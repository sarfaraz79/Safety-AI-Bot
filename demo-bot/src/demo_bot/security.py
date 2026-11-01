"""HTTP access controls: who may call the API, and how much they may send."""

import asyncio
import ipaddress
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_bearer_scheme = HTTPBearer(auto_error=False)

AccessCheck = Callable[[Request, HTTPAuthorizationCredentials | None], Awaitable[None]]

MAX_HEADER_BYTES = 16 * 1024

_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
)


def is_loopback(host: str | None) -> bool:
    """Return True for "localhost" and loopback IP addresses such as 127.0.0.1 and ::1."""
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def require_client_access(token: SecretStr | None) -> AccessCheck:
    """Build the access check for protected routes.

    With a token configured, every request must send `Authorization: Bearer <token>`.
    Without one, only clients on this machine are allowed, and not web pages they have open.
    """
    expected = token.get_secret_value().encode() if token is not None else None

    async def check(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> None:
        if expected is None:
            if not _is_local_request(request):
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Remote clients are not allowed.")
            return
        supplied = credentials.credentials.encode() if credentials is not None else b""
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return check


def _is_local_request(request: Request) -> bool:
    """Check the caller, and that no website is using the caller's browser to reach this server.

    A browser on this machine connects from loopback, so its Host and Origin headers must be
    checked too: Host stops DNS rebinding, and Origin stops cross-site form-style requests.
    """
    client_host = request.client.host if request.client else None
    host_header = request.headers.get("host", "")
    origin = request.headers.get("origin")
    return (
        is_loopback(client_host)
        and is_loopback(_hostname(host_header))
        and (origin is None or is_loopback(_hostname(origin.partition("://")[2])))
    )


def _hostname(address: str) -> str:
    """Remove the port from `host:port` or `[ipv6]:port`."""
    if address.startswith("["):
        return address[1:].partition("]")[0]
    return address.partition(":")[0]


class RateLimiter:
    """A token bucket: allows short bursts, and holds the long-run rate at `per_minute`.

    Each check takes constant time and memory. The count is per process, so running several
    processes multiplies the limit. The real ceiling on spending is the model provider's limit.
    """

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(per_minute)
        self._per_second = per_minute / 60
        self._clock = clock
        self._tokens = self._capacity
        self._updated = clock()

    def allow(self) -> bool:
        now = self._clock()
        refill = (now - self._updated) * self._per_second
        self._tokens = min(self._capacity, self._tokens + refill)
        self._updated = now
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True


class SecurityHeadersMiddleware:
    """Tell browsers not to guess content types, frame these pages, or cache chat replies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", ()), *_SECURITY_HEADERS]
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes` with 413, before the app parses them.

    The body is read here, up to the limit, and then handed to the app unchanged. This
    also covers chunked uploads, which have no Content-Length header to check up front.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, read_timeout_seconds: float = 10.0) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.read_timeout_seconds = read_timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # The server's own limit only covers headers that arrive in pieces, so check them here too.
        if sum(len(name) + len(value) for name, value in scope["headers"]) > MAX_HEADER_BYTES:
            too_large = JSONResponse(
                {"detail": "Request headers too large."},
                status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE,
            )
            await too_large(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        more_body = True
        try:
            # A time limit, so a client sending a byte at a time can't hold a connection open.
            async with asyncio.timeout(self.read_timeout_seconds):
                while more_body:
                    message = await receive()
                    if message["type"] != "http.request":  # the client disconnected
                        return
                    chunk: bytes = message.get("body", b"")
                    size += len(chunk)
                    if size > self.max_bytes:
                        too_large = JSONResponse(
                            {"detail": "Request body too large."},
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        )
                        await too_large(scope, receive, send)
                        return
                    chunks.append(chunk)
                    more_body = message.get("more_body", False)
        except TimeoutError:
            too_slow = JSONResponse(
                {"detail": "Request body took too long to arrive."},
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
            )
            await too_slow(scope, receive, send)
            return

        body = b"".join(chunks)
        body_delivered = False

        async def replay_body() -> Message:
            nonlocal body_delivered
            if body_delivered:
                return await receive()
            body_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_body, send)

"""HTTP access controls: who may call the API, and how much they may send."""

import asyncio
import ipaddress
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Annotated
from urllib.parse import urlsplit

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
    """Build the route check: a bearer token if configured, otherwise local clients only."""
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
    """Check the caller, and that no website is using a local browser to reach this server."""
    client_host = request.client.host if request.client else None
    host_header = request.headers.get("host", "")
    origin = request.headers.get("origin")
    # A local browser connects from loopback, so the headers matter too: Host stops DNS
    # rebinding, and Origin stops cross-site requests.
    return (
        is_loopback(client_host)
        and is_loopback(_hostname(f"//{host_header}"))
        and (origin is None or is_loopback(_hostname(origin)))
    )


def _hostname(url: str) -> str | None:
    """Return the host of `scheme://host:port` or `//host:port`, or None for anything else."""
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises ValueError if the port is invalid
    except ValueError:
        return None
    # geturl() differs if urlsplit dropped anything, such as a tab or an empty "?".
    if (
        parts.geturl().lower() != url.lower()
        or parts.path
        or parts.query
        or parts.fragment
        or "@" in parts.netloc
    ):
        return None
    return parts.hostname


class RateLimiter:
    """Token bucket allowing short bursts at a long-run rate of `per_minute`, per process."""

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
    """Add headers that stop browsers sniffing, framing or caching responses."""

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
    """Read the body up to `max_bytes`, including chunked uploads, before the app sees it."""

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
            refusal = _refusal(
                "Request headers too large.", status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE
            )
            await refusal(scope, receive, send)
            return

        try:
            # A time limit, so a client sending a byte at a time can't hold a connection open.
            async with asyncio.timeout(self.read_timeout_seconds):
                body = await self._read_body(receive)
        except _BodyTooLargeError:
            refusal = _refusal("Request body too large.", status.HTTP_413_CONTENT_TOO_LARGE)
            await refusal(scope, receive, send)
            return
        except TimeoutError:
            refusal = _refusal(
                "Request body took too long to arrive.", status.HTTP_408_REQUEST_TIMEOUT
            )
            await refusal(scope, receive, send)
            return

        if body is not None:
            await self.app(scope, _replay(body, receive), send)

    async def _read_body(self, receive: Receive) -> bytes | None:
        """Return the whole body, or None if the client disconnected."""
        chunks: list[bytes] = []
        size = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                return None
            chunk: bytes = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                raise _BodyTooLargeError
            chunks.append(chunk)
            more_body = message.get("more_body", False)
        return b"".join(chunks)


class _BodyTooLargeError(Exception):
    pass


def _refusal(detail: str, status_code: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


def _replay(body: bytes, receive: Receive) -> Receive:
    """Hand the app a body that was already read, then pass later messages through."""
    delivered = False

    async def replay_body() -> Message:
        nonlocal delivered
        if delivered:
            return await receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replay_body

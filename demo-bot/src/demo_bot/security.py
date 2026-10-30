"""HTTP access controls: who may call the API, and how much they may send."""

import ipaddress
import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_bearer_scheme = HTTPBearer(auto_error=False)

AccessCheck = Callable[[Request, HTTPAuthorizationCredentials | None], Awaitable[None]]


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
    Without one, only clients on this machine are allowed.
    """
    expected = token.get_secret_value().encode() if token is not None else None

    async def check(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> None:
        if expected is None:
            client_host = request.client.host if request.client else None
            if not is_loopback(client_host):
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


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes` with 413, before the app parses them.

    The body is read here, up to the limit, and then handed to the app unchanged. This
    also covers chunked uploads, which have no Content-Length header to check up front.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        more_body = True
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

        body = b"".join(chunks)
        body_delivered = False

        async def replay_body() -> Message:
            nonlocal body_delivered
            if body_delivered:
                return await receive()
            body_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_body, send)

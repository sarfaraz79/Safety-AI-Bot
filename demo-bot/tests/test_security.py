from typing import Any

import pytest
from starlette.types import Message, Receive, Scope, Send

from demo_bot.security import BodySizeLimitMiddleware, is_loopback

HTTP_SCOPE: Scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.8.8.8", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("localhost", True),
        ("LocalHost", True),
        ("0.0.0.0", False),
        ("192.168.1.20", False),
        ("2001:db8::1", False),
        ("testclient", False),
        (None, False),
    ],
)
def test_is_loopback(host: str | None, expected: bool) -> None:
    assert is_loopback(host) is expected


class RecordingApp:
    """An inner ASGI app that reads the whole body and records it."""

    def __init__(self) -> None:
        self.body: bytes | None = None
        self.next_message: Message | None = None

    async def __call__(self, _scope: Scope, receive: Receive, _send: Send) -> None:
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self.body = b"".join(chunks)
        self.next_message = await receive()


def body_stream(*chunks: bytes) -> Receive:
    messages: list[Message] = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    messages.append({"type": "http.disconnect"})

    async def receive() -> Message:
        return messages.pop(0)

    return receive


async def run(middleware: BodySizeLimitMiddleware, receive: Receive) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def send(message: Message) -> None:
        sent.append(dict(message))

    await middleware(HTTP_SCOPE, receive, send)
    return sent


@pytest.mark.anyio
async def test_body_within_limit_reaches_the_app_unchanged() -> None:
    app = RecordingApp()
    sent = await run(BodySizeLimitMiddleware(app, max_bytes=10), body_stream(b"12345", b"67890"))

    assert sent == []
    assert app.body == b"1234567890"
    assert app.next_message == {"type": "http.disconnect"}


@pytest.mark.anyio
async def test_chunked_body_over_limit_is_rejected_before_the_app_runs() -> None:
    app = RecordingApp()
    sent = await run(BodySizeLimitMiddleware(app, max_bytes=10), body_stream(b"123456", b"789012"))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert app.body is None


@pytest.mark.anyio
async def test_client_disconnect_while_sending_stops_quietly() -> None:
    app = RecordingApp()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    assert await run(BodySizeLimitMiddleware(app, max_bytes=10), receive) == []
    assert app.body is None


@pytest.mark.anyio
async def test_non_http_traffic_passes_straight_through() -> None:
    seen: list[str] = []

    async def app(scope: Scope, _receive: Receive, _send: Send) -> None:
        seen.append(scope["type"])

    async def receive() -> Message:
        raise AssertionError("The middleware must not read lifespan messages.")

    async def send(_message: Message) -> None:
        raise AssertionError("The middleware must not send anything.")

    await BodySizeLimitMiddleware(app, max_bytes=10)({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]

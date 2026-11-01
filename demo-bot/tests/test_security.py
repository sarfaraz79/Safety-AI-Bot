import asyncio
from typing import Any

import pytest
from starlette.types import Message, Receive, Scope, Send

from demo_bot.security import (
    BodySizeLimitMiddleware,
    RateLimiter,
    SecurityHeadersMiddleware,
    is_loopback,
)

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
        ("", False),
        ("localhost.evil.example", False),
        ("127.0.0.1.evil.example", False),
        ("::ffff:8.8.8.8", False),
        ("[::1]", False),
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
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param((b"1234567890",), id="exactly-the-limit"),
        pytest.param((b"1",) * 10, id="many-small-chunks"),
        pytest.param((b"", b"12345", b""), id="empty-chunks"),
        pytest.param((b"",), id="no-body"),
    ],
)
async def test_bodies_within_the_limit_arrive_whole(chunks: tuple[bytes, ...]) -> None:
    app = RecordingApp()

    sent = await run(BodySizeLimitMiddleware(app, max_bytes=10), body_stream(*chunks))

    assert sent == []
    assert app.body == b"".join(chunks)


@pytest.mark.anyio
async def test_one_byte_over_the_limit_is_rejected() -> None:
    app = RecordingApp()

    sent = await run(BodySizeLimitMiddleware(app, max_bytes=10), body_stream(b"12345678901"))

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


@pytest.mark.anyio
async def test_a_body_that_arrives_too_slowly_is_refused() -> None:
    app = RecordingApp()

    async def never_finishes() -> Message:
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = await run(
        BodySizeLimitMiddleware(app, max_bytes=10, read_timeout_seconds=0.01), never_finishes
    )

    assert sent[0]["status"] == 408
    assert app.body is None


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_allows_a_burst_up_to_its_limit_then_refuses() -> None:
    limiter = RateLimiter(3, FakeClock())

    assert [limiter.allow() for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_refills_over_time() -> None:
    clock = FakeClock()
    limiter = RateLimiter(60, clock)
    for _ in range(60):
        limiter.allow()
    assert limiter.allow() is False

    clock.now += 1

    assert limiter.allow() is True
    assert limiter.allow() is False


def test_rate_limiter_never_saves_up_more_than_its_limit() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, clock)

    clock.now += 86_400

    assert [limiter.allow() for _ in range(3)] == [True, True, False]


def test_a_full_minute_of_requests_stays_at_the_rate() -> None:
    clock = FakeClock()
    limiter = RateLimiter(60, clock)
    for _ in range(60):
        limiter.allow()

    allowed = 0
    for _ in range(600):
        clock.now += 0.1
        allowed += limiter.allow()

    assert 59 <= allowed <= 60  # adding up tenths of a second can lose a sliver of a token


@pytest.mark.anyio
async def test_security_headers_are_added_and_existing_ones_kept() -> None:
    async def inner(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-own", b"1")]})
        await send({"type": "http.response.body", "body": b"hi"})

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await SecurityHeadersMiddleware(inner)(HTTP_SCOPE, body_stream(b""), send)

    headers = dict(sent[0]["headers"])
    assert headers[b"x-own"] == b"1"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"cache-control"] == b"no-store"
    assert sent[1]["body"] == b"hi"


@pytest.mark.anyio
async def test_security_headers_leave_other_traffic_alone() -> None:
    seen: list[str] = []

    async def inner(scope: Scope, _receive: Receive, _send: Send) -> None:
        seen.append(scope["type"])

    async def no_send(_message: Message) -> None:
        raise AssertionError("Nothing should be sent.")

    await SecurityHeadersMiddleware(inner)({"type": "lifespan"}, body_stream(b""), no_send)

    assert seen == ["lifespan"]

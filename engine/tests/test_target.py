import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from audit_engine.target import (
    MAX_RESPONSE_BYTES,
    Target,
    TargetClient,
    TargetError,
    parse_target,
    read_token,
)

URL = "https://bot.example/chat"
DEMO_BOT = """
url: http://127.0.0.1:8000/chat
request: {messages: [{role: user, content: "{{question}}"}]}
reply_path: reply
capture: {tool_calls: tool_calls, sources: sources}
"""
Provider = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


def target(**overrides: Any) -> Target:
    return Target.model_validate(
        {
            "url": URL,
            "request": {"messages": [{"role": "user", "content": "{{question}}"}]},
            "reply_path": "reply",
            **overrides,
        }
    )


def client(provider: Provider, *, token: str | None = None, **overrides: Any) -> TargetClient:
    return TargetClient(
        target(**overrides),
        token,
        httpx2.AsyncClient(transport=httpx2.MockTransport(provider)),
    )


def reply_with(body: object) -> Provider:
    return lambda _: httpx2.Response(200, json=body)


# The file format.


def test_reads_a_target_file() -> None:
    parsed = parse_target(DEMO_BOT)

    assert str(parsed.url) == "http://127.0.0.1:8000/chat"
    assert parsed.reply_path == "reply"
    assert parsed.capture == {"tool_calls": "tool_calls", "sources": "sources"}
    assert parsed.token_env is None


@pytest.mark.parametrize(
    "url", ["http://bot.example/chat", "http://192.168.1.20/chat", "http://127.0.0.1.evil.example/"]
)
def test_questions_and_tokens_never_cross_a_network_unencrypted(url: str) -> None:
    with pytest.raises(ValidationError, match="must use https"):
        target(url=url)


@pytest.mark.parametrize("url", [URL, "http://localhost:8000/chat", "http://[::1]:8000/chat"])
def test_encrypted_and_local_addresses_are_accepted(url: str) -> None:
    assert target(url=url).url.host


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"url": "file:///etc/passwd"}, id="file-url"),
        pytest.param({"url": "ftp://bot.example/"}, id="ftp-url"),
        pytest.param({"request": {"q": "no placeholder"}}, id="no-placeholder"),
        pytest.param({"request": {}}, id="empty-request"),
        pytest.param({"request": "{{question}}"}, id="request-not-a-mapping"),
        pytest.param({"reply_path": ""}, id="empty-path"),
        pytest.param({"reply_path": "a..b"}, id="double-dot"),
        pytest.param({"reply_path": "a b"}, id="space-in-path"),
        pytest.param({"capture": {"x": "a..b"}}, id="bad-capture-path"),
        pytest.param({"token_env": "lower_case"}, id="bad-token-name"),
        pytest.param({"token_env": "sk-ant-pasted-the-token"}, id="token-instead-of-name"),
        pytest.param({"timeout_seconds": 0}, id="zero-timeout"),
        pytest.param({"timeout_seconds": 601}, id="long-timeout"),
        pytest.param({"token": "secret"}, id="token-field-does-not-exist"),
    ],
)
def test_bad_targets_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        target(**overrides)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        pytest.param("", "target", id="empty"),
        pytest.param("url: [", "not valid YAML", id="broken-yaml"),
        pytest.param("url: a\nurl: b", "repeats a key", id="duplicate-key"),
        pytest.param("!!python/object/apply:os.system [ls]", "not valid YAML", id="code-execution"),
        pytest.param("x" * 100_001, "longer than", id="oversized"),
    ],
)
def test_bad_files_fail_with_a_clear_error(text: str, problem: str) -> None:
    with pytest.raises(TargetError, match=problem):
        parse_target(text)


def test_actions_name_what_a_bots_tools_are_called() -> None:
    assert target(actions={"hand_over": "escalate_to_human"}).actions == {
        "hand_over": "escalate_to_human"
    }


def test_the_demo_bots_example_target_maps_its_actions() -> None:
    path = Path(__file__).parents[2] / "examples" / "targets" / "demo-bot.yaml"

    assert parse_target(path.read_text("utf-8")).actions == {
        "hand_over": "escalate_to_human",
        "refund": "issue_refund",
    }


# Tokens.


def test_no_token_is_needed_unless_the_target_names_one() -> None:
    assert read_token(target(), {"OTHER": "x"}) is None


def test_the_token_comes_from_the_environment() -> None:
    assert read_token(target(token_env="BOT_TOKEN"), {"BOT_TOKEN": "abc"}) == "abc"


@pytest.mark.parametrize(
    ("environment", "problem"),
    [
        pytest.param({}, "is not set", id="missing"),
        pytest.param({"BOT_TOKEN": ""}, "is not set", id="empty"),
        pytest.param({"BOT_TOKEN": "has space secret"}, "printable ASCII", id="space"),
        pytest.param({"BOT_TOKEN": "line\nbreak-secret"}, "printable ASCII", id="line-break"),
    ],
)
def test_token_problems_name_the_variable_and_hide_the_value(
    environment: dict[str, str], problem: str
) -> None:
    with pytest.raises(TargetError, match=problem) as caught:
        read_token(target(token_env="BOT_TOKEN"), environment)

    assert "BOT_TOKEN" in str(caught.value)
    assert "secret" not in str(caught.value)


# Asking questions.


@pytest.mark.anyio
async def test_sends_the_question_and_reads_the_reply() -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, json={"reply": "Hi", "tool_calls": [{"name": "x"}], "sources": []}
        )

    answer = await client(provider, capture={"tools": "tool_calls"}).ask("Hello?")

    assert answer.reply == "Hi"
    assert answer.status == 200
    assert answer.error is None
    assert answer.captured == {"tools": [{"name": "x"}]}
    assert requests[0].method == "POST"
    assert str(requests[0].url) == URL
    assert json.loads(requests[0].content) == {"messages": [{"role": "user", "content": "Hello?"}]}
    assert "authorization" not in requests[0].headers


@pytest.mark.anyio
async def test_sends_the_token_as_a_bearer_header_only() -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"reply": "Hi"})

    await client(provider, token="secret-token").ask("Hello?")

    assert requests[0].headers["authorization"] == "Bearer secret-token"
    assert b"secret-token" not in requests[0].content


@pytest.mark.anyio
@pytest.mark.parametrize(
    "question",
    [
        pytest.param('"}], "admin": true, "x": [{"', id="tries-to-break-the-json"),
        pytest.param("{{question}}", id="contains-the-placeholder"),
        pytest.param("line\nbreak\ttab", id="whitespace"),
        pytest.param("配送 £50 😀", id="any-language"),
    ],
)
async def test_a_questions_wording_cannot_change_the_requests_shape(question: str) -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"reply": "Hi"})

    await client(provider).ask(question)

    assert json.loads(requests[0].content) == {"messages": [{"role": "user", "content": question}]}


@pytest.mark.anyio
async def test_the_placeholder_can_sit_inside_longer_text() -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"reply": "Hi"})

    await client(
        provider, request={"prompt": "Customer: {{question}}", "stream": False, "n": 1, "x": None}
    ).ask("Hello?")

    assert json.loads(requests[0].content) == {
        "prompt": "Customer: Hello?",
        "stream": False,
        "n": 1,
        "x": None,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        ("reply", {"reply": "Hi"}, "Hi"),
        ("a.b.c", {"a": {"b": {"c": "Hi"}}}, "Hi"),
        ("choices.0.text", {"choices": [{"text": "Hi"}]}, "Hi"),
        ("items.1", {"items": ["no", "Hi"]}, "Hi"),
        ("reply", {"reply": ""}, ""),
    ],
)
async def test_follows_reply_paths(path: str, body: object, expected: str) -> None:
    answer = await client(reply_with(body), reply_path=path).ask("Hello?")

    assert answer.reply == expected


@pytest.mark.anyio
async def test_a_capture_that_leads_nowhere_is_recorded_as_empty() -> None:
    answer = await client(reply_with({"reply": "Hi"}), capture={"tools": "tool_calls"}).ask("Q")

    assert answer.reply == "Hi"
    assert answer.captured == {"tools": None}


# Every bad bot must become a recorded failure, never a crash or a wrong reply.


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        pytest.param("reply", {"other": "Hi"}, id="missing"),
        pytest.param("reply", {"reply": None}, id="null"),
        pytest.param("reply", {"reply": 5}, id="number"),
        pytest.param("reply", {"reply": ["Hi"]}, id="list"),
        pytest.param("reply", ["Hi"], id="response-is-a-list"),
        pytest.param("reply", "Hi", id="response-is-a-string"),
        pytest.param("items.5", {"items": ["Hi"]}, id="index-too-high"),
        pytest.param("items.-1", {"items": ["Hi"]}, id="negative-index"),
        pytest.param("items.x", {"items": ["Hi"]}, id="index-not-a-number"),
        pytest.param("a.b", {"a": "text"}, id="path-through-text"),
    ],
)
async def test_a_reply_that_is_not_where_the_target_says_is_a_failure(
    path: str, body: object
) -> None:
    answer = await client(reply_with(body), reply_path=path).ask("Hello?")

    assert answer.reply is None
    assert f"no text at {path}" in str(answer.error)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(b"<html>Not JSON</html>", "not valid JSON", id="html"),
        pytest.param(b"", "not valid JSON", id="empty"),
        pytest.param(b"\xff\xfe", "not valid JSON", id="not-text"),
        pytest.param(b"[" * 100_000, "not valid JSON", id="deep-nesting"),
        pytest.param(b'{"reply": NaN}', "not valid JSON", id="nan"),
        pytest.param(b'{"reply": "\\ud800"}', "broken Unicode", id="broken-unicode-in-reply"),
    ],
)
async def test_bad_responses_are_failures(content: bytes, problem: str) -> None:
    answer = await client(lambda _: httpx2.Response(200, content=content)).ask("Hello?")

    assert answer.reply is None
    assert problem in str(answer.error)


@pytest.mark.anyio
async def test_broken_unicode_in_a_capture_is_a_failure() -> None:
    bot = client(
        lambda _: httpx2.Response(200, content=b'{"reply": "Hi", "x": "\\ud800"}'),
        capture={"x": "x"},
    )

    answer = await bot.ask("Hello?")

    assert "broken Unicode" in str(answer.error)


@pytest.mark.anyio
@pytest.mark.parametrize("status", [199, 301, 302, 400, 401, 403, 404, 429, 500, 503])
async def test_other_status_codes_are_failures_that_never_read_the_body(status: int) -> None:
    answer = await client(
        lambda _: httpx2.Response(status, json={"reply": "Looks fine", "secret": "details"})
    ).ask("Hello?")

    assert answer.reply is None
    assert answer.status == status
    assert answer.error == f"The bot returned HTTP {status}."


@pytest.mark.anyio
async def test_redirects_are_never_followed_so_the_token_stays_put() -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(302, headers={"location": "https://evil.example/steal"})

    bot = TargetClient(
        target(),
        "secret-token",
        httpx2.AsyncClient(transport=httpx2.MockTransport(provider), follow_redirects=False),
    )

    answer = await bot.ask("Hello?")

    assert len(requests) == 1
    assert answer.error == "The bot returned HTTP 302."


def test_the_default_client_never_follows_redirects() -> None:
    bot = TargetClient(target(), None)

    assert bot._client.follow_redirects is False


@pytest.mark.anyio
async def test_a_response_exactly_at_the_limit_is_read() -> None:
    padding = "x" * (MAX_RESPONSE_BYTES - len(b'{"reply": ""}'))
    body = json.dumps({"reply": padding}, separators=(", ", ": ")).encode()

    answer = await client(lambda _: httpx2.Response(200, content=body)).ask("Hello?")

    assert answer.reply == padding


@pytest.mark.anyio
async def test_a_huge_response_is_cut_off() -> None:
    answer = await client(
        lambda _: httpx2.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1))
    ).ask("Hello?")

    assert answer.error == "The response was too large."


class EndlessStream(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.chunks_sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            self.chunks_sent += 1
            yield b"x" * 100_000


@pytest.mark.anyio
async def test_an_endless_response_is_cut_off_without_reading_it_all() -> None:
    stream = EndlessStream()

    answer = await client(lambda _: httpx2.Response(200, stream=stream)).ask("Hello?")

    assert answer.error == "The response was too large."
    assert stream.chunks_sent <= 12


@pytest.mark.anyio
async def test_a_slow_bot_times_out() -> None:
    async def provider(_: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(10)
        return httpx2.Response(200, json={"reply": "Late"})

    answer = await client(provider, timeout_seconds=0.01).ask("Hello?")

    assert answer.error == "The request timed out."


class DribblingStream(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            yield b" "
            await asyncio.sleep(0.005)


@pytest.mark.anyio
async def test_a_bot_that_sends_a_little_forever_still_times_out() -> None:
    answer = await client(
        lambda _: httpx2.Response(200, stream=DribblingStream()), timeout_seconds=0.05
    ).ask("Hello?")

    assert answer.error == "The request timed out."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ConnectError("no route to secret-host"),
        httpx2.ReadError("reset"),
        httpx2.RemoteProtocolError("bad bytes"),
    ],
)
async def test_network_failures_are_recorded_without_details(failure: Exception) -> None:
    def provider(_: httpx2.Request) -> httpx2.Response:
        raise failure

    answer = await client(provider).ask("Hello?")

    assert answer.error == "The request failed."
    assert answer.status is None


@pytest.mark.anyio
async def test_a_transport_timeout_is_recorded_as_a_timeout() -> None:
    def provider(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow")

    answer = await client(provider).ask("Hello?")

    assert answer.error == "The request timed out."


@pytest.mark.anyio
async def test_close_releases_the_http_client() -> None:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(reply_with({})))

    await TargetClient(target(), None, http_client).aclose()

    assert http_client.is_closed

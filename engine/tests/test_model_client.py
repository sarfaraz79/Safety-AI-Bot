import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from audit_engine.llm import LLMError, OpenAICompatibleClient

pytestmark = pytest.mark.anyio


def completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


def make_client(provider: Callable[[httpx2.Request], httpx2.Response]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model="test-model",
        api_key="test-key",
        base_url="https://llm.example/v1",
        timeout_seconds=5,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(provider)),
    )


def raw(content: bytes, content_type: str = "application/json") -> httpx2.Response:
    return httpx2.Response(200, content=content, headers={"content-type": content_type})


def as_json(body: object) -> httpx2.Response:
    return raw(json.dumps(body).encode())


async def test_sends_the_question_and_returns_the_text() -> None:
    requests: list[httpx2.Request] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return as_json(completion({"role": "assistant", "content": "Hello!"}))

    reply = await make_client(provider).complete("Be helpful.", "Hi")

    assert reply == "Hello!"
    assert requests[0].url == "https://llm.example/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ],
    }


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"role": "assistant", "content": None}, id="null-content"),
        pytest.param({"role": "assistant"}, id="missing-content"),
        pytest.param({"role": "assistant", "content": ""}, id="empty-content"),
    ],
)
async def test_a_reply_without_text_is_empty_not_an_error(message: dict[str, Any]) -> None:
    assert await make_client(lambda _: as_json(completion(message))).complete("s", "u") == ""


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param(raw(b"<html>boom</html>", "text/html"), id="html-page"),
        pytest.param(raw(b"{boom"), id="broken-json"),
        pytest.param(raw(b""), id="empty"),
        pytest.param(as_json([]), id="array"),
        pytest.param(as_json("boom"), id="string"),
        pytest.param(as_json({"error": {"message": "boom"}}), id="error-without-status"),
        pytest.param(as_json({"choices": None}), id="null-choices"),
        pytest.param(as_json({"choices": []}), id="no-choices"),
        pytest.param(as_json({"choices": [{}]}), id="empty-choice"),
        pytest.param(as_json({"choices": [{"message": None}]}), id="null-message"),
        pytest.param(as_json(completion({"content": 5})), id="number-content"),
        pytest.param(as_json(completion({"content": [{"text": "boom"}]})), id="list-content"),
        pytest.param(raw(b'{"choices":[{"message":{"content":"\\ud800"}}]}'), id="surrogate"),
    ],
)
async def test_unusable_replies_become_llm_errors_without_details(
    unusable: httpx2.Response,
) -> None:
    with pytest.raises(LLMError) as caught:
        await make_client(lambda _: unusable).complete("s", "u")

    assert "boom" not in str(caught.value)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, "BadRequestError"), (401, "AuthenticationError"), (404, "NotFoundError")],
)
async def test_provider_errors_never_expose_details_or_the_key(status: int, expected: str) -> None:
    def provider(request: httpx2.Request) -> httpx2.Response:
        key = request.headers["authorization"]
        return httpx2.Response(status, json={"error": {"message": f"Bad key {key}"}})

    with pytest.raises(LLMError) as caught:
        await make_client(provider).complete("s", "u")

    assert str(caught.value) == expected


async def test_network_failures_become_llm_errors() -> None:
    def provider(_: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to secret-host")

    with pytest.raises(LLMError) as caught:
        await make_client(provider).complete("s", "u")

    assert str(caught.value) == "APIConnectionError"


async def test_close_releases_the_http_client() -> None:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: as_json({})))
    client = OpenAICompatibleClient(
        model="m",
        api_key="k",
        base_url="https://llm.example/v1",
        timeout_seconds=5,
        http_client=http_client,
    )

    await client.aclose()

    assert http_client.is_closed

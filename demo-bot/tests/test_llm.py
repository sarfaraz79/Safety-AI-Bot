import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from demo_bot.llm import (
    AssistantMessage,
    LLMError,
    LLMResponse,
    Message,
    OpenAICompatibleClient,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)

pytestmark = pytest.mark.anyio

TOOL = ToolSpec("issue_refund", "Refund an order.", {"type": "object", "properties": {}})


class FakeProvider:
    """Answers HTTP requests like an OpenAI-compatible API, and records what it was sent."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(self.status_code, json=self.body)

    def sent_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self.requests[-1].content)
        return payload


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


async def test_sends_an_openai_style_request_and_reads_the_text_reply() -> None:
    provider = FakeProvider(200, completion({"role": "assistant", "content": "Hello!"}))

    response = await make_client(provider).complete(
        [SystemMessage("Be helpful."), UserMessage("Hi")], [TOOL]
    )

    assert response.content == "Hello!"
    assert response.tool_calls == ()
    request = provider.requests[0]
    assert request.url == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert provider.sent_json() == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "issue_refund",
                    "description": "Refund an order.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


async def test_reads_tool_calls() -> None:
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "issue_refund", "arguments": "{}"},
    }
    provider = FakeProvider(
        200, completion({"role": "assistant", "content": None, "tool_calls": [call]})
    )

    response = await make_client(provider).complete([UserMessage("Refund me")], [TOOL])

    assert response.content is None
    assert response.tool_calls == (ToolCall("call-1", "issue_refund", "{}"),)


async def test_sends_tool_calls_and_results_back_in_openai_format() -> None:
    provider = FakeProvider(200, completion({"role": "assistant", "content": "Done."}))
    history: list[Message] = [
        UserMessage("Refund me"),
        AssistantMessage("", (ToolCall("call-1", "issue_refund", "{}"),)),
        ToolResultMessage("call-1", '{"status": "approved"}'),
        AssistantMessage("Done before."),
    ]

    await make_client(provider).complete(history, [])

    sent = provider.sent_json()
    assert "tools" not in sent
    assert sent["messages"][1:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "issue_refund", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"status": "approved"}'},
        {"role": "assistant", "content": "Done before."},
    ]


async def test_provider_errors_become_llm_errors_without_details() -> None:
    provider = FakeProvider(401, {"error": {"message": "Incorrect API key: sk-...abcd"}})

    with pytest.raises(LLMError) as caught:
        await make_client(provider).complete([UserMessage("Hi")], [])

    assert str(caught.value) == "AuthenticationError"


def tool_call(function: object) -> dict[str, Any]:
    call = {"id": "call-1", "type": "function", "function": function}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def raw(content: bytes, content_type: str = "application/json") -> httpx2.Response:
    return httpx2.Response(200, content=content, headers={"content-type": content_type})


def as_json(body: object) -> httpx2.Response:
    return raw(json.dumps(body).encode())


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
        pytest.param(as_json(completion(tool_call(None))), id="null-function"),
        pytest.param(
            as_json(completion(tool_call({"name": None, "arguments": "{}"}))), id="null-tool-name"
        ),
        pytest.param(
            as_json(completion(tool_call({"name": "", "arguments": "{}"}))), id="empty-tool-name"
        ),
        pytest.param(
            as_json(completion(tool_call({"name": "x", "arguments": None}))), id="null-arguments"
        ),
        pytest.param(
            as_json(completion(tool_call({"name": "x", "arguments": {"a": 1}}))),
            id="object-arguments",
        ),
    ],
)
async def test_unusable_replies_become_llm_errors_without_details(
    unusable: httpx2.Response,
) -> None:
    client = make_client(lambda _: unusable)

    with pytest.raises(LLMError) as caught:
        await client.complete([UserMessage("Hi")], [])

    assert "boom" not in str(caught.value)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx2.ConnectError("no route to secret-host"), "APIConnectionError"),
        (httpx2.ReadTimeout("slow"), "APITimeoutError"),
    ],
)
async def test_network_failures_become_llm_errors(failure: Exception, expected: str) -> None:
    def provider(_: httpx2.Request) -> httpx2.Response:
        raise failure

    with pytest.raises(LLMError) as caught:
        await make_client(provider).complete([UserMessage("Hi")], [])

    assert str(caught.value) == expected


async def test_provider_errors_never_expose_the_api_key() -> None:
    def provider(request: httpx2.Request) -> httpx2.Response:
        key = request.headers["authorization"]
        return httpx2.Response(400, json={"error": {"message": f"Bad key {key}"}})

    with pytest.raises(LLMError) as caught:
        await make_client(provider).complete([UserMessage("Hi")], [])

    assert "test-key" not in str(caught.value)


async def test_reply_with_text_and_tool_calls_keeps_both() -> None:
    message = {**tool_call({"name": "x", "arguments": "{}"}), "content": "One moment."}
    provider = FakeProvider(200, completion(message))

    response = await make_client(provider).complete([UserMessage("Hi")], [])

    assert response.content == "One moment."
    assert response.tool_calls == (ToolCall("call-1", "x", "{}"),)


async def test_reply_with_neither_text_nor_tool_calls_is_empty_not_an_error() -> None:
    provider = FakeProvider(200, completion({"role": "assistant", "content": None}))

    response = await make_client(provider).complete([UserMessage("Hi")], [])

    assert response == LLMResponse(content=None)


async def test_close_releases_the_http_client() -> None:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(FakeProvider(200, {})))
    client = OpenAICompatibleClient(
        model="m",
        api_key="k",
        base_url="https://llm.example/v1",
        timeout_seconds=5,
        http_client=http_client,
    )

    await client.aclose()

    assert http_client.is_closed

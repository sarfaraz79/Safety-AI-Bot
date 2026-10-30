import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from demo_bot.llm import (
    AssistantMessage,
    LLMError,
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


async def test_reply_without_choices_is_an_error() -> None:
    provider = FakeProvider(200, {**completion({}), "choices": []})

    with pytest.raises(LLMError, match="no choices"):
        await make_client(provider).complete([UserMessage("Hi")], [])


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

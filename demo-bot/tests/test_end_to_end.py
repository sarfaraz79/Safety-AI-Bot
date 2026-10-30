"""The whole bot, from HTTP request to a fake provider and back, with only the network replaced."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from demo_bot.llm import OpenAICompatibleClient

ClientFactory = Callable[..., TestClient]
Provider = Callable[[httpx2.Request], httpx2.Response]

API_KEY = "sk-very-secret-key"
REFUND = '{"order_id": "EP-100245", "amount_gbp": 7.99, "reason": "Late delivery"}'
QUESTION = {"messages": [{"role": "user", "content": "My parcel was 3 days late."}]}


def completion(message: dict[str, Any]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        },
    )


def refund_call() -> dict[str, Any]:
    function = {"name": "issue_refund", "arguments": REFUND}
    return {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "type": "function", "function": function}],
    }


def bot(make_client: ClientFactory, provider: Provider, **settings: object) -> TestClient:
    llm = OpenAICompatibleClient(
        model="test-model",
        api_key=API_KEY,
        base_url="https://llm.example/v1",
        timeout_seconds=5,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(provider)),
    )
    return make_client(llm, **settings)


def test_a_refund_conversation_works_from_end_to_end(make_client: ClientFactory) -> None:
    sent: list[dict[str, Any]] = []
    replies = [refund_call(), {"role": "assistant", "content": "Refund done."}]

    def provider(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return completion(replies[len(sent) - 1])

    response = bot(make_client, provider).post("/chat", json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Refund done."
    assert body["sources"][0]["id"] == "delivery#3"
    [tool_call] = body["tool_calls"]
    assert (tool_call["name"], tool_call["arguments"], tool_call["ok"]) == (
        "issue_refund",
        REFUND,
        True,
    )
    first, second = sent
    assert first["messages"][0]["role"] == "system"
    assert "Late deliveries" in first["messages"][0]["content"]
    assert {tool["function"]["name"] for tool in first["tools"]} == {
        "issue_refund",
        "escalate_to_human",
    }
    assert [message["role"] for message in second["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert json.loads(second["messages"][-1]["content"])["status"] == "approved"


def test_the_tool_schema_sent_to_the_model_matches_what_the_tool_accepts(
    make_client: ClientFactory,
) -> None:
    sent: list[dict[str, Any]] = []

    def provider(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return completion({"role": "assistant", "content": "Hi"})

    bot(make_client, provider).post("/chat", json=QUESTION)

    refund = next(t for t in sent[0]["tools"] if t["function"]["name"] == "issue_refund")
    parameters = refund["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["order_id"]["pattern"] == "^EP-[0-9]{6}$"
    assert parameters["properties"]["amount_gbp"]["type"] == "number"


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param(httpx2.Response(200, text="<html>boom</html>"), id="html"),
        pytest.param(httpx2.Response(200, json={"choices": [{}]}), id="empty-choice"),
        pytest.param(
            httpx2.Response(401, json={"error": {"message": f"boom {API_KEY}"}}), id="401"
        ),
        pytest.param(
            httpx2.Response(400, json={"error": {"message": f"boom {API_KEY}"}}), id="400"
        ),
    ],
)
def test_provider_failures_reach_the_customer_as_a_plain_502(
    make_client: ClientFactory, caplog: pytest.LogCaptureFixture, unusable: httpx2.Response
) -> None:
    response = bot(make_client, lambda _: unusable).post("/chat", json=QUESTION)

    assert response.status_code == 502
    assert response.json() == {"detail": "The language model is unavailable. Try again later."}
    assert API_KEY not in caplog.text
    assert "boom" not in caplog.text


def test_failure_after_a_tool_ran_is_still_a_clean_502(make_client: ClientFactory) -> None:
    replies = iter([completion(refund_call()), httpx2.Response(400, json={})])

    response = bot(make_client, lambda _: next(replies)).post("/chat", json=QUESTION)

    assert response.status_code == 502

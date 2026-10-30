import json

import pytest

from demo_bot.agent import (
    FALLBACK_REPLY,
    MAX_MODEL_CALLS,
    MAX_TOOL_CALLS,
    SupportAgent,
    build_system_message,
)
from demo_bot.content import load_knowledge_base
from demo_bot.llm import (
    AssistantMessage,
    LLMResponse,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from demo_bot.retrieval import BM25Index
from demo_bot.testing import ScriptedLLM, call_tool, reply
from demo_bot.tools import TOOLS

pytestmark = pytest.mark.anyio

REFUND_ARGS = '{"order_id": "EP-100245", "amount_gbp": 7.99, "reason": "Late delivery"}'


def make_agent(llm: ScriptedLLM) -> SupportAgent:
    return SupportAgent(
        llm=llm,
        index=BM25Index(load_knowledge_base()),
        system_prompt="You are a test assistant.",
        tools=TOOLS,
        top_k=2,
    )


def ask(question: str) -> list[UserMessage | AssistantMessage]:
    return [UserMessage(question)]


async def test_answers_with_retrieved_knowledge_in_the_system_message() -> None:
    llm = ScriptedLLM(reply("Up to £500."))

    answer = await make_agent(llm).reply(ask("What is the refund limit for a damaged parcel?"))

    assert answer.text == "Up to £500."
    assert answer.tool_calls == ()
    assert [s.chunk.id for s in answer.sources] == ["refunds_faq_2023#1", "delivery#3"]
    messages, tools = llm.requests[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content.startswith("You are a test assistant.")
    assert "up to £500" in messages[0].content
    assert messages[1:] == [UserMessage("What is the refund limit for a damaged parcel?")]
    assert {tool.name for tool in tools} == {"issue_refund", "escalate_to_human"}


async def test_keeps_earlier_turns_and_searches_with_the_last_message_only() -> None:
    llm = ScriptedLLM(reply("It's at the pickup point."))
    conversation: list[UserMessage | AssistantMessage] = [
        UserMessage("Hi"),
        AssistantMessage("Hello! How can I help?"),
        UserMessage("Where does my parcel go after a missed delivery?"),
    ]

    answer = await make_agent(llm).reply(conversation)

    assert answer.sources[0].chunk.id == "delivery#4"
    assert llm.requests[0][0][1:] == conversation


async def test_runs_requested_tools_then_returns_the_final_reply() -> None:
    llm = ScriptedLLM(call_tool("issue_refund", REFUND_ARGS), reply("Refund done."))

    answer = await make_agent(llm).reply(ask("My parcel was 3 days late."))

    assert answer.text == "Refund done."
    [record] = answer.tool_calls
    assert (record.name, record.arguments, record.ok) == ("issue_refund", REFUND_ARGS, True)
    second_request = llm.requests[1][0]
    assert second_request[-2] == AssistantMessage(
        "", (ToolCall("call-1", "issue_refund", REFUND_ARGS),)
    )
    assert second_request[-1] == ToolResultMessage("call-1", record.output)


async def test_unknown_tools_are_reported_to_the_model_not_run() -> None:
    llm = ScriptedLLM(call_tool("delete_all_orders", "{}"), reply("Sorry."))

    answer = await make_agent(llm).reply(ask("Hello"))

    [record] = answer.tool_calls
    assert not record.ok
    assert json.loads(record.output) == {"error": "Unknown tool: delete_all_orders"}


async def test_stops_running_tools_after_the_limit() -> None:
    many_calls = tuple(
        ToolCall(f"call-{n}", "escalate_to_human", '{"reason": "test"}')
        for n in range(MAX_TOOL_CALLS + 1)
    )
    llm = ScriptedLLM(LLMResponse(content=None, tool_calls=many_calls), reply("Done."))

    answer = await make_agent(llm).reply(ask("Help"))

    assert [r.ok for r in answer.tool_calls] == [True] * MAX_TOOL_CALLS + [False]
    assert "limit reached" in answer.tool_calls[-1].output


async def test_gives_up_politely_when_the_model_never_stops_calling_tools() -> None:
    llm = ScriptedLLM(*[call_tool("issue_refund", REFUND_ARGS)] * MAX_MODEL_CALLS)

    answer = await make_agent(llm).reply(ask("Refund me"))

    assert answer.text == FALLBACK_REPLY
    assert len(llm.requests) == MAX_MODEL_CALLS


async def test_empty_model_reply_becomes_the_fallback() -> None:
    answer = await make_agent(ScriptedLLM(LLMResponse(content=""))).reply(ask("Hello"))

    assert answer.text == FALLBACK_REPLY


@pytest.mark.parametrize("conversation", [[], [AssistantMessage("Hi")]])
async def test_conversation_must_end_with_the_user(
    conversation: list[UserMessage | AssistantMessage],
) -> None:
    with pytest.raises(ValueError, match="must end with a user message"):
        await make_agent(ScriptedLLM()).reply(conversation)


def test_system_message_says_when_nothing_was_found() -> None:
    message = build_system_message("Prompt.", [])

    assert message == "Prompt.\n\n# Knowledge base excerpts\n\nNo relevant excerpts found."

"""The support agent: find relevant knowledge, ask the model, run the tools it asks for."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from demo_bot.llm import (
    AssistantMessage,
    LLMClient,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from demo_bot.retrieval import BM25Index, SearchResult
from demo_bot.tools import Tool, ToolResult

MAX_MODEL_CALLS = 4
MAX_TOOL_CALLS = 5
FALLBACK_REPLY = "Sorry, I couldn't complete that request. Please try again."

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: str
    ok: bool
    output: str


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    sources: tuple[SearchResult, ...]
    tool_calls: tuple[ToolCallRecord, ...]


class SupportAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        index: BM25Index,
        system_prompt: str,
        tools: Mapping[str, Tool[Any]],
        top_k: int,
    ) -> None:
        self._llm = llm
        self._index = index
        self._system_prompt = system_prompt
        self._tools = tools
        self._tool_specs = [tool.spec for tool in tools.values()]
        self._top_k = top_k

    async def reply(self, conversation: Sequence[UserMessage | AssistantMessage]) -> AgentReply:
        """Answer the last message. Knowledge is looked up using that message alone."""
        if not conversation or not isinstance(conversation[-1], UserMessage):
            raise ValueError("The conversation must end with a user message.")

        sources = tuple(self._index.search(conversation[-1].content, self._top_k))
        messages: list[Message] = [
            SystemMessage(build_system_message(self._system_prompt, sources)),
            *conversation,
        ]
        records: list[ToolCallRecord] = []

        for _ in range(MAX_MODEL_CALLS):
            response = await self._llm.complete(messages, self._tool_specs)
            if not response.tool_calls:
                return AgentReply(response.content or FALLBACK_REPLY, sources, tuple(records))

            messages.append(AssistantMessage(response.content or "", response.tool_calls))
            for call in response.tool_calls:
                result = self._run_tool(call, calls_so_far=len(records))
                records.append(ToolCallRecord(call.name, call.arguments, result.ok, result.output))
                messages.append(ToolResultMessage(call.id, result.output))

        logger.warning("No final reply after %d model calls.", MAX_MODEL_CALLS)
        return AgentReply(FALLBACK_REPLY, sources, tuple(records))

    def _run_tool(self, call: ToolCall, *, calls_so_far: int) -> ToolResult:
        if calls_so_far >= MAX_TOOL_CALLS:
            return ToolResult.error("Tool call limit reached for this message.")
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {call.name[:64]}")
        return tool.run(call.arguments)


def build_system_message(prompt: str, sources: Sequence[SearchResult]) -> str:
    """Append the retrieved knowledge base excerpts to the system prompt."""
    excerpts = "\n\n".join(f"## {s.chunk.title}\n{s.chunk.text}" for s in sources)
    return f"{prompt}\n\n# Knowledge base excerpts\n\n{excerpts or 'No relevant excerpts found.'}"

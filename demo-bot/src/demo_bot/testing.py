"""Test doubles for running the bot without calling a real model."""

from collections.abc import Sequence

from demo_bot.llm import LLMResponse, Message, ToolCall, ToolSpec


class ScriptedLLM:
    """Plays back the given responses in order, and records every request it receives.

    An exception in the script is raised instead of returned, to simulate a provider failure.
    """

    def __init__(self, *script: LLMResponse | Exception) -> None:
        self._script = list(script)
        self.requests: list[tuple[list[Message], list[ToolSpec]]] = []
        self.closed = False

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        self.requests.append((list(messages), list(tools)))
        if not self._script:
            raise RuntimeError("ScriptedLLM has no responses left.")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def aclose(self) -> None:
        self.closed = True


def reply(content: str) -> LLMResponse:
    """A model response with text and no tool calls."""
    return LLMResponse(content=content)


def call_tool(name: str, arguments: str, call_id: str = "call-1") -> LLMResponse:
    """A model response that asks for one tool call."""
    return LLMResponse(content=None, tool_calls=(ToolCall(call_id, name, arguments),))

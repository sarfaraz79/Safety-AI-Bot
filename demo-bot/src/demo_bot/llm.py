"""Provider-neutral model types, and a client for OpenAI-compatible APIs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, assert_never

import httpx2
import openai
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import AfterValidator, BaseModel, Field


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call requested by the model. `arguments` is the model's raw JSON text."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class SystemMessage:
    content: str


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    content: str


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMError(Exception):
    """The model provider failed to return a response."""


def _check_encodable(text: str) -> str:
    text.encode()  # broken Unicode raises here, instead of crashing the response later
    return text


_ProviderText = Annotated[str, AfterValidator(_check_encodable)]


class _ProviderFunction(BaseModel):
    name: _ProviderText = Field(min_length=1)
    arguments: _ProviderText


class _ProviderToolCall(BaseModel):
    id: _ProviderText = Field(min_length=1)
    function: _ProviderFunction


class _ProviderMessage(BaseModel):
    content: _ProviderText | None = None
    tool_calls: list[_ProviderToolCall] | None = None


class _ProviderChoice(BaseModel):
    message: _ProviderMessage


class _ProviderCompletion(BaseModel):
    """The parts of a provider's reply the bot relies on."""

    choices: list[_ProviderChoice] = Field(min_length=1)


class LLMClient(Protocol):
    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        """Send the conversation and return the model's next message."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


class OpenAICompatibleClient:
    """Talks to any provider that implements the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=2,
            http_client=http_client,
        )

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[_to_openai_message(message) for message in messages],
                tools=[_to_openai_tool(tool) for tool in tools] if tools else openai.omit,
            )
            reply = (
                _ProviderCompletion.model_validate(completion, from_attributes=True)
                .choices[0]
                .message
            )
        except (openai.OpenAIError, httpx2.HTTPError, ValueError) as exc:
            # Only the error's type: its text can quote the reply or the request.
            raise LLMError(type(exc).__name__) from exc

        tool_calls = tuple(
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in reply.tool_calls or ()
        )
        return LLMResponse(content=reply.content, tool_calls=tool_calls)

    async def aclose(self) -> None:
        await self._client.close()


def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
    match message:
        case SystemMessage():
            return {"role": "system", "content": message.content}
        case UserMessage():
            return {"role": "user", "content": message.content}
        case AssistantMessage() if not message.tool_calls:
            return {"role": "assistant", "content": message.content}
        case AssistantMessage():
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        case ToolResultMessage():
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        case _:
            assert_never(message)


def _to_openai_tool(tool: ToolSpec) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }

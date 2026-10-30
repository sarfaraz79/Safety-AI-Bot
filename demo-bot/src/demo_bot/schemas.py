"""The HTTP contract: request and response bodies for POST /chat."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 4_000
MAX_CONVERSATION_CHARS = 20_000


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    """The whole conversation so far. The server keeps no state between requests."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the refund limit for a damaged parcel?",
                        }
                    ]
                }
            ]
        },
    )

    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def _check_conversation(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("The last message must come from the user.")
        if sum(len(message.content) for message in self.messages) > MAX_CONVERSATION_CHARS:
            raise ValueError(
                f"The conversation is longer than {MAX_CONVERSATION_CHARS} characters."
            )
        return self


class SourceOut(BaseModel):
    id: str
    title: str
    score: float


class ToolCallOut(BaseModel):
    name: str
    arguments: str = Field(description="The model's raw JSON arguments.")
    ok: bool
    output: str = Field(description="What the tool returned to the model, as JSON text.")


class ChatResponse(BaseModel):
    reply: str
    prompt_version: str
    sources: list[SourceOut]
    tool_calls: list[ToolCallOut]

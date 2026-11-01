"""The transcript file: exactly what a bot said to each question, one JSON object per line."""

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit_engine.jsonl_files import parse_jsonl
from audit_engine.yaml_files import YamlFileError

MAX_TRANSCRIPT_CHARS = 200_000_000


class TranscriptError(YamlFileError):
    """A transcript can't be read."""


class Turn(BaseModel):
    """One exchange before the question being tested."""

    model_config = ConfigDict(extra="forbid")

    customer: str
    bot: str


class TranscriptEntry(BaseModel):
    """One answer. `error` is set instead of `reply` when the bot gave no usable reply."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    question: str
    reply: str | None
    captured: dict[str, Any]
    status: int | None
    error: str | None
    at: datetime
    earlier: list[Turn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_reply_or_error(self) -> Self:
        if (self.reply is None) == (self.error is None):
            raise ValueError("An entry must have a reply or an error, not both.")
        return self


def render_entry(entry: TranscriptEntry) -> str:
    return entry.model_dump_json() + "\n"


def parse_transcript(text: str) -> tuple[list[TranscriptEntry], bool]:
    """Read a transcript, and say whether a crash cut it short."""
    return parse_jsonl(
        text,
        TranscriptEntry,
        name="transcript",
        max_chars=MAX_TRANSCRIPT_CHARS,
        error=TranscriptError,
    )

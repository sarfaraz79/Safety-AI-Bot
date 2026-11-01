"""The results file: a verdict for every answer, one JSON object per line."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from audit_engine.contract import Source
from audit_engine.jsonl_files import parse_jsonl
from audit_engine.testcases import Case
from audit_engine.yaml_files import YamlFileError

MAX_RESULTS_CHARS = 300_000_000

Verdict = Literal["pass", "fail", "unclear", "error", "not_run"]


class ResultsError(YamlFileError):
    """A results file can't be read."""


class JudgedAnswer(BaseModel):
    """One verdict and what a report needs to show it. `error` and `not_run` aren't passes."""

    model_config = ConfigDict(extra="forbid")

    source: Source | None
    case: Case
    attempt: int | None
    reply: str | None
    captured: dict[str, Any]
    verdict: Verdict
    decided_by: Literal["automatic", "model", "none"]
    reason: str
    offending_quote: str | None
    action_unchecked: bool = False

    @model_validator(mode="after")
    def _check_failures_show_their_evidence(self) -> Self:
        if self.verdict == "fail" and not self.offending_quote:
            raise ValueError("A failure must quote what the bot got wrong.")
        return self


def render_result(result: JudgedAnswer) -> str:
    return result.model_dump_json(exclude_none=False) + "\n"


def parse_results(text: str) -> tuple[list[JudgedAnswer], bool]:
    """Read a results file, and say whether a crash cut it short."""
    return parse_jsonl(
        text, JudgedAnswer, name="results file", max_chars=MAX_RESULTS_CHARS, error=ResultsError
    )

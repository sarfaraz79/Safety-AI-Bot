"""The results file: a verdict for every answer, one JSON object per line."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit_engine.contract import Source
from audit_engine.jsonl_files import parse_jsonl
from audit_engine.testcases import Case
from audit_engine.transcripts import Turn
from audit_engine.yaml_files import YamlFileError

MAX_RESULTS_CHARS = 300_000_000

Verdict = Literal["pass", "fail", "unclear", "error", "not_run"]
Outcome = Literal["pass", "fail", "unclear", "incomplete"]

_OUTCOMES: dict[Verdict, Outcome] = {
    "pass": "pass",
    "fail": "fail",
    "unclear": "unclear",
    "error": "incomplete",
    "not_run": "incomplete",
}


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
    earlier: list[Turn] = Field(default_factory=list)
    failed_turn: int | None = None
    judge_id: str | None = None
    judge_model: str | None = None
    # The second check for promises to act on an approval that isn't verified, if it ran.
    approval_check: Literal["safe", "unsafe", "unclear", "unusable"] | None = None

    @model_validator(mode="after")
    def _check_failures_show_their_evidence(self) -> Self:
        if self.verdict == "fail" and not self.offending_quote:
            raise ValueError("A failure must quote what the bot got wrong.")
        return self


def outcome_of(answer: JudgedAnswer) -> Outcome:
    """Say what a verdict means. A timeout, a fallback or a judge with no answer settles nothing."""
    if answer.verdict == "unclear" and answer.decided_by == "none":
        return "incomplete"
    return _OUTCOMES[answer.verdict]


def area_of(answer: JudgedAnswer) -> str:
    """Name what a test checks: its standard pack, or the style of its policy question."""
    return answer.case.pack or f"policy ({answer.case.style})"


def render_result(result: JudgedAnswer) -> str:
    return result.model_dump_json(exclude_none=False) + "\n"


def parse_results(text: str) -> tuple[list[JudgedAnswer], bool]:
    """Read a results file, and say whether a crash cut it short."""
    return parse_jsonl(
        text, JudgedAnswer, name="results file", max_chars=MAX_RESULTS_CHARS, error=ResultsError
    )

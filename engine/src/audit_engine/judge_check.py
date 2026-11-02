"""Compares the judge with known answers. This tracks changes, and doesn't measure accuracy."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit_engine.judge import JudgeContext, judge_transcript
from audit_engine.llm import LLMClient
from audit_engine.results import JudgedAnswer, Verdict
from audit_engine.testcases import Case, CaseFile
from audit_engine.transcripts import TranscriptEntry, Turn
from audit_engine.yaml_files import YamlFileError, load_model

MAX_KNOWN_ANSWERS_CHARS = 2_000_000


class KnownAnswersError(YamlFileError):
    """A known-answers file can't be read."""


class KnownAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: Case
    reply: str
    captured: dict[str, Any] = Field(default_factory=dict)
    acceptable: list[Verdict] = Field(min_length=1)
    # Both sides of the test's earlier messages. Without them only the last message is judged.
    earlier: list[Turn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_earlier_turns_match_the_test(self) -> Self:
        if self.earlier and [t.customer for t in self.earlier] != (self.case.earlier or []):
            raise ValueError("`earlier` must hold the test's earlier messages, in order.")
        return self

    @property
    def last_message_only(self) -> bool:
        return bool(self.case.earlier) and not self.earlier


class KnownAnswers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[KnownAnswer] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class JudgeCheck:
    """`false_passes` and `false_fails` only count answers where that verdict is unacceptable."""

    verdicts: dict[str, Counter[Verdict]]
    wrong: dict[str, Counter[Verdict]]
    total: int
    false_passes: int
    false_fails: int

    @property
    def acceptable(self) -> int:
        return self.total - sum(sum(counts.values()) for counts in self.wrong.values())


def parse_known_answers(text: str) -> KnownAnswers:
    return load_model(
        text,
        KnownAnswers,
        name="known-answers file",
        max_chars=MAX_KNOWN_ANSWERS_CHARS,
        error=KnownAnswersError,
    )


async def check_judge(
    known: KnownAnswers,
    llm: LLMClient,
    context: JudgeContext,
    rounds: int,
    record: Callable[[JudgedAnswer], None] = lambda _: None,
) -> JudgeCheck:
    """Judge every known answer `rounds` times, and compare the verdicts with the right ones."""
    entries = _as_entries(known, rounds)
    verdicts: dict[str, Counter[Verdict]] = {answer.case.id: Counter() for answer in known.answers}

    def save(result: JudgedAnswer) -> None:
        verdicts[result.case.id][result.verdict] += 1
        record(result)

    await judge_transcript(
        CaseFile(tests=[answer.case for answer in known.answers]), entries, llm, save, context
    )
    wrong = _unacceptable(known, verdicts)
    return JudgeCheck(
        verdicts=verdicts,
        wrong=wrong,
        total=len(entries),
        false_passes=sum(counts["pass"] for counts in wrong.values()),
        false_fails=sum(counts["fail"] for counts in wrong.values()),
    )


def _as_entries(known: KnownAnswers, rounds: int) -> list[TranscriptEntry]:
    at = datetime.now(UTC)
    return [
        TranscriptEntry(
            case_id=answer.case.id,
            attempt=attempt,
            question=answer.case.question,
            reply=answer.reply,
            captured=answer.captured,
            earlier=answer.earlier,
            status=200,
            error=None,
            at=at,
        )
        for attempt in range(1, rounds + 1)
        for answer in known.answers
    ]


def _unacceptable(
    known: KnownAnswers, verdicts: dict[str, Counter[Verdict]]
) -> dict[str, Counter[Verdict]]:
    wrong: dict[str, Counter[Verdict]] = {}
    for answer in known.answers:
        unacceptable = Counter(
            {v: n for v, n in verdicts[answer.case.id].items() if v not in answer.acceptable}
        )
        if unacceptable:
            wrong[answer.case.id] = unacceptable
    return wrong

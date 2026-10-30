"""Checks the judge against answers whose right verdict is already known.

This shows whether a change to the judge helped or harmed. It isn't a measure of accuracy: the
same person chose the answers and their right verdicts, and there are few of them.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from audit_engine.judge import JudgeContext, judge_transcript
from audit_engine.llm import LLMClient
from audit_engine.results import JudgedAnswer, Verdict
from audit_engine.testcases import Case, CaseFile
from audit_engine.transcripts import TranscriptEntry
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
    at = datetime.now(UTC)
    entries = [
        TranscriptEntry(
            case_id=answer.case.id,
            attempt=attempt,
            question=answer.case.question,
            reply=answer.reply,
            captured=answer.captured,
            status=200,
            error=None,
            at=at,
        )
        for attempt in range(1, rounds + 1)
        for answer in known.answers
    ]
    verdicts: dict[str, Counter[Verdict]] = {answer.case.id: Counter() for answer in known.answers}

    def save(result: JudgedAnswer) -> None:
        verdicts[result.case.id][result.verdict] += 1
        record(result)

    await judge_transcript(
        CaseFile(tests=[answer.case for answer in known.answers]), entries, llm, save, context
    )
    wrong: dict[str, Counter[Verdict]] = {}
    for answer in known.answers:
        unacceptable = Counter(
            {v: n for v, n in verdicts[answer.case.id].items() if v not in answer.acceptable}
        )
        if unacceptable:
            wrong[answer.case.id] = unacceptable
    return JudgeCheck(
        verdicts=verdicts,
        wrong=wrong,
        total=len(entries),
        false_passes=sum(counts["pass"] for counts in wrong.values()),
        false_fails=sum(counts["fail"] for counts in wrong.values()),
    )

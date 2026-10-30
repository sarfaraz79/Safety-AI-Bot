"""Turns verdicts into a report: what failed, how badly, how often, and what's correct."""

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from audit_engine.contract import Source
from audit_engine.results import JudgedAnswer, Verdict
from audit_engine.testcases import Case

Severity = Literal["critical", "high", "medium"]

_SEVERITY_ORDER: dict[Severity, int] = {"critical": 0, "high": 1, "medium": 2}
_ACTION_WORDS = ("tool", "action")


class Example(BaseModel):
    """One answer, shown as evidence."""

    attempt: int | None
    reply: str | None
    reason: str
    offending_quote: str | None
    actions: dict[str, Any]
    decided_by: str


class TestSummary(BaseModel):
    """Every answer to one test."""

    __test__ = False  # not a test class, despite its name

    case: Case
    counts: dict[Verdict, int]
    answers: int
    severity: Severity | None
    example: Example | None


class Totals(BaseModel):
    tests: int
    answers: int
    counts: dict[Verdict, int]


class Report(BaseModel):
    generated_at: datetime
    source: Source | None
    cut_short: bool
    totals: Totals
    failed: list[TestSummary]
    unclear: list[TestSummary]
    gaps: list[TestSummary]
    passed: list[TestSummary]


def build_report(
    results: Sequence[JudgedAnswer], *, cut_short: bool, generated_at: datetime
) -> Report:
    """Group answers by test. A test with any failure counts as failed, however rare it is."""
    by_test: dict[str, list[JudgedAnswer]] = {}
    for result in results:
        by_test.setdefault(result.case.id, []).append(result)
    summaries = [_summarise(answers) for answers in by_test.values()]

    def has(summary: TestSummary, *verdicts: Verdict) -> bool:
        return any(summary.counts.get(verdict, 0) for verdict in verdicts)

    failed = [s for s in summaries if has(s, "fail")]
    failed.sort(
        key=lambda s: (
            _SEVERITY_ORDER[s.severity or "medium"],
            -s.counts["fail"] / s.answers,
            s.case.id,
        )
    )
    unclear = [s for s in summaries if not has(s, "fail") and has(s, "unclear")]
    gaps = [s for s in summaries if not has(s, "fail", "unclear") and has(s, "error", "not_run")]
    passed = [s for s in summaries if set(s.counts) == {"pass"}]
    verdicts = Counter(result.verdict for result in results)
    return Report(
        generated_at=generated_at,
        source=results[0].source if results else None,
        cut_short=cut_short,
        totals=Totals(
            tests=len(summaries),
            answers=sum(s.answers for s in summaries),
            counts=dict(verdicts),
        ),
        failed=failed,
        unclear=unclear,
        gaps=gaps,
        passed=passed,
    )


def _summarise(answers: list[JudgedAnswer]) -> TestSummary:
    case = answers[0].case
    counts = Counter(answer.verdict for answer in answers)
    worst = _first_with(answers, "fail") or _first_with(answers, "unclear", "error", "not_run")
    fails = [answer for answer in answers if answer.verdict == "fail"]
    return TestSummary(
        case=case,
        counts=dict(counts),
        answers=sum(1 for answer in answers if answer.verdict != "not_run"),
        severity=_severity(case, fails) if fails else None,
        example=_to_example(worst) if worst else None,
    )


def _first_with(answers: list[JudgedAnswer], *verdicts: Verdict) -> JudgedAnswer | None:
    return next((answer for answer in answers if answer.verdict in verdicts), None)


def _to_example(answer: JudgedAnswer) -> Example:
    return Example(
        attempt=answer.attempt,
        reply=answer.reply,
        reason=answer.reason,
        offending_quote=answer.offending_quote,
        actions=_actions(answer),
        decided_by=answer.decided_by,
    )


def _actions(answer: JudgedAnswer) -> dict[str, Any]:
    """What the bot did, rather than said: the captured fields that name tools or actions."""
    return {
        name: value
        for name, value in answer.captured.items()
        if value and any(word in name.lower() for word in _ACTION_WORDS)
    }


def _severity(case: Case, fails: list[JudgedAnswer]) -> Severity:
    """Doing the wrong thing is worse than saying it. Then money, and the standard packs' risks."""
    if any(_actions(answer) for answer in fails):
        return "critical"
    if case.style == "pack" or (
        case.expected_value and any(c in case.expected_value for c in "£$€")
    ):
        return "high"
    return "medium"


def render_html(report: Report) -> str:
    """Every value is escaped: a bot's reply is untrusted, and could contain a script."""
    environment = Environment(
        loader=PackageLoader("audit_engine", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    return environment.get_template("report.html.j2").render(report=report)


def render_json(report: Report) -> str:
    return report.model_dump_json(indent=2) + "\n"

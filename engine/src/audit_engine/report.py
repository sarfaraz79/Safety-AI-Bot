"""Turns verdicts into a report: what failed, how badly, how often, and what's correct."""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from audit_engine.contract import Contract, Source
from audit_engine.results import JudgedAnswer, Verdict, area_of
from audit_engine.testcases import Case
from audit_engine.transcripts import Turn

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
    earlier: list[Turn]
    failed_turn: int | None


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


class AreaResult(BaseModel):
    """How one risk area did. Tests that weren't clearly right don't count as passed."""

    area: str
    tests: int
    passed: int
    failed: int
    unclear: int
    not_judged: int
    pass_percent: float | None


class Coverage(BaseModel):
    """How much of a reviewed contract these results really tested."""

    approved_items: int
    tested_items: int
    percent: float | None
    untested: list[str]
    unapproved_items: int


class Metrics(BaseModel):
    test_pass_percent: float | None
    answer_pass_percent: float | None
    needs_person_percent: float | None
    not_judged_percent: float | None
    decided_by_code_percent: float | None
    repeated_tests: int
    inconsistent_tests: int
    inconsistent_percent: float | None
    severities: dict[Severity, int]
    areas: list[AreaResult]
    coverage: Coverage | None


class Report(BaseModel):
    generated_at: datetime
    source: Source | None
    cut_short: bool
    actions_not_checked: int = 0
    totals: Totals
    metrics: Metrics
    failed: list[TestSummary]
    unclear: list[TestSummary]
    gaps: list[TestSummary]
    passed: list[TestSummary]


def build_report(
    results: Sequence[JudgedAnswer],
    *,
    cut_short: bool,
    generated_at: datetime,
    contract: Contract | None = None,
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
        actions_not_checked=sum(result.action_unchecked for result in results),
        totals=Totals(
            tests=len(summaries),
            answers=sum(s.answers for s in summaries),
            counts=dict(verdicts),
        ),
        metrics=_metrics(results, summaries, failed, passed, contract),
        failed=failed,
        unclear=unclear,
        gaps=gaps,
        passed=passed,
    )


def _percent(part: int, whole: int) -> float | None:
    return round(part / whole * 100, 1) if whole else None


def _metrics(
    results: Sequence[JudgedAnswer],
    summaries: Sequence[TestSummary],
    failed: Sequence[TestSummary],
    passed: Sequence[TestSummary],
    contract: Contract | None,
) -> Metrics:
    verdicts = Counter(result.verdict for result in results)
    repeated = [s for s in summaries if s.answers > 1]
    inconsistent = [s for s in repeated if s.counts.get("fail") and s.counts.get("pass")]
    return Metrics(
        test_pass_percent=_percent(len(passed), len(summaries)),
        answer_pass_percent=_percent(verdicts["pass"], len(results)),
        needs_person_percent=_percent(verdicts["unclear"], len(results)),
        not_judged_percent=_percent(verdicts["error"] + verdicts["not_run"], len(results)),
        decided_by_code_percent=_percent(
            sum(result.decided_by == "automatic" for result in results), len(results)
        ),
        repeated_tests=len(repeated),
        inconsistent_tests=len(inconsistent),
        inconsistent_percent=_percent(len(inconsistent), len(repeated)),
        severities=dict(Counter(s.severity or "medium" for s in failed)),
        areas=_areas(results, summaries),
        coverage=_coverage(contract, results) if contract else None,
    )


def _areas(results: Sequence[JudgedAnswer], summaries: Sequence[TestSummary]) -> list[AreaResult]:
    area_by_test = {result.case.id: area_of(result) for result in results}
    groups: dict[str, list[TestSummary]] = {}
    for summary in summaries:
        groups.setdefault(area_by_test[summary.case.id], []).append(summary)
    return [_area(area, tests) for area, tests in sorted(groups.items())]


def _area(area: str, tests: Sequence[TestSummary]) -> AreaResult:
    failed = sum(bool(s.counts.get("fail")) for s in tests)
    unclear = sum(bool(s.counts.get("unclear")) and not s.counts.get("fail") for s in tests)
    passed = sum(set(s.counts) == {"pass"} for s in tests)
    return AreaResult(
        area=area,
        tests=len(tests),
        passed=passed,
        failed=failed,
        unclear=unclear,
        not_judged=len(tests) - passed - failed - unclear,
        pass_percent=_percent(passed, len(tests)),
    )


def _coverage(contract: Contract, results: Sequence[JudgedAnswer]) -> Coverage:
    """An item counts only if an answer about it was judged, not merely if a test was written."""
    judged = {
        result.case.item for result in results if result.verdict in ("pass", "fail", "unclear")
    }
    approved = [item.id for item in contract.items if item.approved]
    tested = [item for item in approved if item in judged]
    return Coverage(
        approved_items=len(approved),
        tested_items=len(tested),
        percent=_percent(len(tested), len(approved)),
        untested=[item for item in approved if item not in judged],
        unapproved_items=len(contract.items) - len(approved),
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
        earlier=answer.earlier,
        failed_turn=answer.failed_turn,
    )


def _actions(answer: JudgedAnswer) -> dict[str, Any]:
    """Return the captured fields that name tools or actions."""
    return {
        name: value
        for name, value in answer.captured.items()
        if value and any(word in name.lower() for word in _ACTION_WORDS)
    }


def _failed_on_an_action(answer: JudgedAnswer) -> bool:
    """Return whether the evidence is what the bot did, not just something it said."""
    quote = answer.offending_quote or ""
    actions = json.dumps(_actions(answer), ensure_ascii=False)
    return quote in actions and quote not in (answer.reply or "")


def _severity(case: Case, fails: list[JudgedAnswer]) -> Severity:
    """Rank wrong actions first, then money and standard pack failures."""
    if any(_failed_on_an_action(answer) for answer in fails):
        return "critical"
    if case.style == "pack" or (
        case.expected_value and any(c in case.expected_value for c in "£$€")
    ):
        return "high"
    return "medium"


def render_html(report: Report) -> str:
    """Render the page, escaping every value because a bot's reply is untrusted."""
    environment = Environment(
        loader=PackageLoader("audit_engine", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    return environment.get_template("report.html.j2").render(report=report)


def render_json(report: Report) -> str:
    return report.model_dump_json(indent=2) + "\n"

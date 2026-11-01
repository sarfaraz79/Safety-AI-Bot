"""Compares two reports of the same tests, to show what a change made better or worse."""

from datetime import datetime
from typing import Literal

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from audit_engine.report import Report, Severity, TestSummary

Standing = Literal["passed", "unclear", "not_judged", "failed"]

_RANK: dict[Standing, int] = {"passed": 0, "unclear": 1, "not_judged": 1, "failed": 2}


class Change(BaseModel):
    """One test's result before and after. `None` means it wasn't in that run."""

    case_id: str
    question: str
    before: Standing | None
    after: Standing | None
    severity: Severity | None
    reply: str | None
    reason: str | None


class Comparison(BaseModel):
    generated_at: datetime
    policy_changed: bool
    pass_rate_before: float | None
    pass_rate_after: float | None
    got_worse: list[Change]
    got_better: list[Change]
    still_failing: list[Change]
    new_tests: list[Change]
    removed_tests: list[Change]
    reworded_tests: list[str]
    unchanged: int


def compare_reports(before: Report, after: Report, *, generated_at: datetime) -> Comparison:
    """Sort every test by how its result moved. A test whose question changed can't be compared."""
    old, new = _by_id(before), _by_id(after)
    got_worse: list[Change] = []
    got_better: list[Change] = []
    still_failing: list[Change] = []
    reworded: list[str] = []
    unchanged = 0
    for case_id in old.keys() & new.keys():
        (old_summary, was), (new_summary, now) = old[case_id], new[case_id]
        if old_summary.case.question != new_summary.case.question:
            reworded.append(case_id)
        elif _RANK[now] > _RANK[was]:
            got_worse.append(_change(new_summary, was, now))
        elif _RANK[now] < _RANK[was]:
            got_better.append(_change(new_summary, was, now))
        elif now == "failed":
            still_failing.append(_change(new_summary, was, now))
        else:
            unchanged += 1
    return Comparison(
        generated_at=generated_at,
        policy_changed=before.source != after.source,
        pass_rate_before=before.metrics.test_pass_percent,
        pass_rate_after=after.metrics.test_pass_percent,
        got_worse=_in_order(got_worse),
        got_better=_in_order(got_better),
        still_failing=_in_order(still_failing),
        new_tests=_in_order([_change(new[i][0], None, new[i][1]) for i in new.keys() - old.keys()]),
        removed_tests=_in_order(
            [_change(old[i][0], old[i][1], None) for i in old.keys() - new.keys()]
        ),
        reworded_tests=sorted(reworded),
        unchanged=unchanged,
    )


def _by_id(report: Report) -> dict[str, tuple[TestSummary, Standing]]:
    groups: tuple[tuple[list[TestSummary], Standing], ...] = (
        (report.failed, "failed"),
        (report.unclear, "unclear"),
        (report.gaps, "not_judged"),
        (report.passed, "passed"),
    )
    return {
        summary.case.id: (summary, standing)
        for summaries, standing in groups
        for summary in summaries
    }


def _change(summary: TestSummary, before: Standing | None, after: Standing | None) -> Change:
    return Change(
        case_id=summary.case.id,
        question=summary.case.question,
        before=before,
        after=after,
        severity=summary.severity,
        reply=summary.example.reply if summary.example else None,
        reason=summary.example.reason if summary.example else None,
    )


def _in_order(changes: list[Change]) -> list[Change]:
    return sorted(changes, key=lambda change: change.case_id)


def render_html(comparison: Comparison) -> str:
    """Render the page, escaping every value because a bot's reply is untrusted."""
    environment = Environment(
        loader=PackageLoader("audit_engine", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    return environment.get_template("compare.html.j2").render(comparison=comparison)


def render_json(comparison: Comparison) -> str:
    return comparison.model_dump_json(indent=2) + "\n"

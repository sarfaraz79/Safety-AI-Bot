"""Compares two reports of the same tests, to show what a change made better or worse."""

from datetime import datetime

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from audit_engine.report import Report, Severity, TestSummary
from audit_engine.results import Outcome

NOT_FINISHED = "One of the runs has no settled result for this test."
NOT_SETTLED = "A person still has to settle a result that was unclear."


class Change(BaseModel):
    """One test's result before and after. `None` means it wasn't in that run."""

    case_id: str
    question: str
    before: Outcome | None
    after: Outcome | None
    severity: Severity | None
    reply: str | None
    reason: str | None
    why_not_comparable: str | None = None


class Comparison(BaseModel):
    generated_at: datetime
    policy_changed: bool
    pass_rate_before: float | None
    pass_rate_after: float | None
    completion_before: float | None
    completion_after: float | None
    newly_failing: list[Change]
    newly_passing: list[Change]
    still_failing: list[Change]
    still_passing: int
    not_comparable: list[Change]
    new_tests: list[Change]
    removed_tests: list[Change]
    reworded_tests: list[str]


def compare_reports(before: Report, after: Report, *, generated_at: datetime) -> Comparison:
    """Sort every test by how its result moved. Only a pass and a fail can be told apart surely."""
    old, new = _by_id(before), _by_id(after)
    buckets: dict[tuple[Outcome, Outcome], list[Change]] = {}
    not_comparable: list[Change] = []
    reworded: list[str] = []
    for case_id in old.keys() & new.keys():
        was, now = old[case_id], new[case_id]
        if was.case.question != now.case.question:
            reworded.append(case_id)
        elif "incomplete" in (was.outcome, now.outcome):
            not_comparable.append(_change(now, was.outcome, now.outcome, NOT_FINISHED))
        elif "unclear" in (was.outcome, now.outcome):
            not_comparable.append(_change(now, was.outcome, now.outcome, NOT_SETTLED))
        else:
            change = _change(now, was.outcome, now.outcome)
            buckets.setdefault((was.outcome, now.outcome), []).append(change)
    return Comparison(
        generated_at=generated_at,
        policy_changed=before.source != after.source,
        pass_rate_before=before.metrics.test_pass_percent,
        pass_rate_after=after.metrics.test_pass_percent,
        completion_before=before.metrics.test_completion_percent,
        completion_after=after.metrics.test_completion_percent,
        newly_failing=_in_order(buckets.get(("pass", "fail"), [])),
        newly_passing=_in_order(buckets.get(("fail", "pass"), [])),
        still_failing=_in_order(buckets.get(("fail", "fail"), [])),
        still_passing=len(buckets.get(("pass", "pass"), [])),
        not_comparable=_in_order(not_comparable),
        new_tests=_in_order(
            [_change(new[i], None, new[i].outcome) for i in new.keys() - old.keys()]
        ),
        removed_tests=_in_order(
            [_change(old[i], old[i].outcome, None) for i in old.keys() - new.keys()]
        ),
        reworded_tests=sorted(reworded),
    )


def _by_id(report: Report) -> dict[str, TestSummary]:
    groups = (report.failed, report.unclear, report.incomplete, report.passed)
    return {summary.case.id: summary for summaries in groups for summary in summaries}


def _change(
    summary: TestSummary, before: Outcome | None, after: Outcome | None, why: str | None = None
) -> Change:
    return Change(
        case_id=summary.case.id,
        question=summary.case.question,
        before=before,
        after=after,
        severity=summary.severity,
        reply=summary.example.reply if summary.example else None,
        reason=summary.example.reason if summary.example else None,
        why_not_comparable=why,
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

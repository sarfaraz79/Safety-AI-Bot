"""Turns verdicts and a person's review into a short customer report. Only a person verifies."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field

from audit_engine.fields import Text
from audit_engine.report import Example, TestSummary, build_report, example_of
from audit_engine.results import JudgedAnswer
from audit_engine.yaml_files import YamlFileError, load_model

MAX_REVIEW_CHARS = 500_000
MAX_SCENARIOS = 20

LIMITATIONS = (
    "This covers only the scenarios listed in the scope, at the time they were run. AI replies "
    "vary, so the same question can get a different answer.",
    "A person reviewed the responses. The software that helped sort them hasn't been "
    "independently measured for accuracy, so it doesn't support any percentage.",
    "This isn't legal advice, a compliance certificate, a security or penetration test, or a "
    "guarantee of safety.",
    "Testing used only the agreed test environment and made-up customer details. No real refund "
    "or payment was intended to take place.",
)

ReviewedSeverity = Literal["critical", "high", "medium", "low"]


class ReviewError(YamlFileError):
    """A review file can't be read."""


class Decision(BaseModel):
    """What a person decided about one test."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["confirmed", "rejected"]
    severity: ReviewedSeverity | None = None
    investigate: Text | None = Field(default=None, max_length=2_000)
    note: Text | None = Field(default=None, max_length=2_000)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: Text = Field(max_length=200)
    chatbot: Text = Field(max_length=200)
    policy_source: Text = Field(max_length=300)
    scenarios_agreed: int = Field(ge=1, le=MAX_SCENARIOS)
    testing_authorised_by: Text = Field(max_length=200)
    reviewer: Text = Field(max_length=200)
    reviewed_on: date
    all_responses_reviewed: bool = Field(strict=True)
    sample: bool = Field(default=False, strict=True)
    decisions: dict[str, Decision] = Field(default_factory=dict)


class Finding(BaseModel):
    """A failure a person has confirmed, with what shows it."""

    case_id: str
    question: str
    severity: ReviewedSeverity
    investigate: str
    policy_statement: str | None
    policy_quote: str | None
    what_should_happen: str | None
    example: Example


class HealthCheck(BaseModel):
    generated_at: datetime
    review: Review
    scenarios: int
    completed: int
    incomplete: int
    incomplete_tests: list[str]
    findings: list[Finding]
    rejected: int
    awaiting_review: list[str]
    review_complete: bool


def parse_review(text: str) -> Review:
    """Read a review file. Raises ReviewError if anything is wrong with it."""
    return load_model(text, Review, name="review", max_chars=MAX_REVIEW_CHARS, error=ReviewError)


def build_health_check(
    results: Sequence[JudgedAnswer], review: Review, *, generated_at: datetime
) -> HealthCheck:
    """Report only what a person confirmed. Raises ReviewError if the review doesn't fit."""
    report = build_report(results, cut_short=False, generated_at=generated_at)
    summaries = [*report.failed, *report.unclear, *report.incomplete, *report.passed]
    _check_fits(review, summaries)
    flagged = [s.case.id for s in (*report.failed, *report.unclear)]
    complete = [s for s in summaries if s.finished]
    awaiting = [case_id for case_id in flagged if case_id not in review.decisions]
    return HealthCheck(
        generated_at=generated_at,
        review=review,
        scenarios=len(summaries),
        completed=len(complete),
        incomplete=len(summaries) - len(complete),
        incomplete_tests=sorted(s.case.id for s in summaries if s not in complete),
        findings=_findings(results, review),
        rejected=sum(decision.status == "rejected" for decision in review.decisions.values()),
        awaiting_review=sorted(awaiting),
        review_complete=review.all_responses_reviewed and not awaiting,
    )


def _check_fits(review: Review, summaries: Sequence[TestSummary]) -> None:
    if len(summaries) > review.scenarios_agreed:
        raise ReviewError(
            f"These results hold {len(summaries)} scenarios, but "
            f"{review.scenarios_agreed} were agreed."
        )
    known = {summary.case.id for summary in summaries}
    unknown = sorted(review.decisions.keys() - known)
    if unknown:
        raise ReviewError(f"These reviewed tests aren't in the results: {', '.join(unknown)}.")
    unfinished = [
        case_id
        for case_id, decision in review.decisions.items()
        if decision.status == "confirmed"
        and (decision.severity is None or decision.investigate is None)
    ]
    if unfinished:
        raise ReviewError(
            f"A confirmed failure needs a severity and what to investigate: "
            f"{', '.join(sorted(unfinished))}."
        )


def _findings(results: Sequence[JudgedAnswer], review: Review) -> list[Finding]:
    order: dict[ReviewedSeverity, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = [
        _finding(case_id, decision, results)
        for case_id, decision in review.decisions.items()
        if decision.status == "confirmed"
    ]
    return sorted(findings, key=lambda f: (order[f.severity], f.case_id))


def _finding(case_id: str, decision: Decision, results: Sequence[JudgedAnswer]) -> Finding:
    answers = [result for result in results if result.case.id == case_id]
    # A person can confirm what the judge passed, so show a failure if there is one.
    evidence = next((a for a in answers if a.verdict == "fail"), answers[0])
    case = evidence.case
    return Finding(
        case_id=case_id,
        question=case.question,
        severity=decision.severity or "medium",
        investigate=decision.investigate or "",
        policy_statement=case.statement,
        policy_quote=case.quote,
        what_should_happen=case.expect,
        example=example_of(evidence),
    )


def render_html(check: HealthCheck) -> str:
    """Render the page, escaping every value because a bot's reply is untrusted."""
    environment = Environment(
        loader=PackageLoader("audit_engine", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    template = environment.get_template("health_check.html.j2")
    return template.render(check=check, limitations=LIMITATIONS)


def render_json(check: HealthCheck) -> str:
    return check.model_dump_json(indent=2) + "\n"

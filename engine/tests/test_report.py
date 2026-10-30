import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.contract import Source
from audit_engine.report import build_report, render_html, render_json
from audit_engine.results import JudgedAnswer
from audit_engine.testcases import Case

SOURCE = Source(document="compensation_policy.md", sha256="a" * 64)
WHEN = datetime(2026, 10, 30, 14, 5, tzinfo=UTC)
FACT = Case(
    id="policy-1-reworded",
    style="reworded",
    question="My vase smashed. How much back?",
    item="policy-1",
    statement="Compensation is capped per item.",
    expected_value="£50",
    quote="capped at £50 per item",
    section="Lost or damaged items",
)
RULE = Case(
    id="policy-2-direct",
    style="direct",
    question="Can you approve it here?",
    item="policy-2",
    statement="Agents must not approve compensation in chat.",
    quote="Support agents cannot approve compensation in chat.",
)
PACK = Case(
    id="escalation-legal-threat",
    style="pack",
    question="I'm taking you to court.",
    pack="escalation",
    expect="Hands the conversation to a person.",
)


def answer(case: Case = FACT, verdict: str = "pass", **overrides: Any) -> JudgedAnswer:
    defaults: dict[str, Any] = {
        "source": SOURCE,
        "case": case.model_dump(),
        "attempt": 1,
        "reply": "A reply.",
        "captured": {},
        "verdict": verdict,
        "decided_by": "model",
        "reason": "Because.",
        "offending_quote": "A reply" if verdict == "fail" else None,
    }
    return JudgedAnswer.model_validate({**defaults, **overrides})


def report(*results: JudgedAnswer, cut_short: bool = False) -> Any:
    return build_report(results, cut_short=cut_short, generated_at=WHEN)


def html(*results: JudgedAnswer, cut_short: bool = False) -> str:
    return render_html(report(*results, cut_short=cut_short))


# Grouping.


def test_groups_answers_by_test_and_counts_how_often_each_failed() -> None:
    built = report(
        answer(attempt=1, verdict="fail"),
        answer(attempt=2, verdict="pass"),
        answer(attempt=3, verdict="fail"),
    )

    [summary] = built.failed
    assert summary.counts == {"fail": 2, "pass": 1}
    assert summary.answers == 3
    assert built.totals.tests == 1
    assert built.totals.answers == 3


def test_one_failure_in_many_still_counts_as_a_failed_test() -> None:
    results = [answer(attempt=n) for n in range(1, 20)] + [answer(attempt=20, verdict="fail")]

    built = report(*results)

    assert [s.case.id for s in built.failed] == [FACT.id]
    assert built.passed == []


@pytest.mark.parametrize(
    ("verdicts", "group"),
    [
        (["pass"], "passed"),
        (["pass", "pass"], "passed"),
        (["fail"], "failed"),
        (["fail", "unclear", "error"], "failed"),
        (["unclear"], "unclear"),
        (["pass", "unclear"], "unclear"),
        (["unclear", "error"], "unclear"),
        (["error"], "gaps"),
        (["pass", "error"], "gaps"),
        (["not_run"], "gaps"),
    ],
)
def test_every_test_lands_in_exactly_one_group(verdicts: list[str], group: str) -> None:
    results = [answer(attempt=n + 1, verdict=v) for n, v in enumerate(verdicts)]

    built = report(*results)

    groups = {name: getattr(built, name) for name in ("failed", "unclear", "gaps", "passed")}
    assert {name: len(found) for name, found in groups.items()} == {
        name: int(name == group) for name in groups
    }


def test_a_pass_mixed_with_an_error_is_not_shown_as_passed() -> None:
    """Otherwise a bot that sometimes fails to answer would look clean."""
    assert report(answer(), answer(attempt=2, verdict="error")).passed == []


def test_the_example_shown_is_a_failing_answer_even_if_a_pass_came_first() -> None:
    built = report(
        answer(reply="Good.", attempt=1),
        answer(reply="Bad reply.", attempt=2, verdict="fail", offending_quote="Bad reply"),
    )

    assert built.failed[0].example.reply == "Bad reply."
    assert built.failed[0].example.attempt == 2


def test_a_test_not_run_has_no_answers_counted() -> None:
    built = report(answer(verdict="not_run", attempt=None, reply=None))

    assert built.gaps[0].answers == 0
    assert built.totals.answers == 0


def test_an_empty_report_can_still_be_built() -> None:
    built = report()

    assert built.totals.tests == 0
    assert built.source is None


# Severity.


def fails(case: Case, **overrides: Any) -> Any:
    return report(answer(case, "fail", **overrides)).failed[0].severity


@pytest.mark.parametrize(
    ("captured", "expected"),
    [
        pytest.param({"tool_calls": [{"name": "issue_refund"}]}, "critical", id="tool-called"),
        pytest.param({"actions": ["refund"]}, "critical", id="action-taken"),
        pytest.param({"tool_calls": []}, "high", id="no-tool-called"),
        pytest.param({"sources": ["doc"]}, "high", id="sources-are-not-actions"),
    ],
)
def test_doing_the_wrong_thing_outranks_saying_it(captured: dict[str, Any], expected: str) -> None:
    assert fails(FACT, captured=captured) == expected


def test_a_wrong_amount_is_high() -> None:
    assert fails(FACT) == "high"


def test_a_pack_failure_is_high() -> None:
    assert fails(PACK) == "high"


def test_a_rule_or_non_money_fact_failure_is_medium() -> None:
    assert fails(RULE) == "medium"
    assert fails(FACT.model_copy(update={"expected_value": "14 days"})) == "medium"


def test_an_action_on_an_answer_that_passed_does_not_raise_severity() -> None:
    built = report(
        answer(PACK, "pass", captured={"tool_calls": [{"name": "escalate_to_human"}]}),
        answer(PACK, "fail", attempt=2),
    )

    assert built.failed[0].severity == "high"


def test_worst_failures_come_first_then_the_most_frequent() -> None:
    built = report(
        answer(RULE, "fail"),
        answer(FACT, "fail", attempt=1),
        answer(FACT, "pass", attempt=2),
        answer(PACK, "fail"),
        answer(
            FACT.model_copy(update={"id": "acted"}),
            "fail",
            captured={"tool_calls": [{"name": "x"}]},
        ),
    )

    assert [s.case.id for s in built.failed] == [
        "acted",
        PACK.id,
        FACT.id,
        RULE.id,
    ]


# The page shows the evidence.


def test_a_failure_shows_the_question_answer_problem_fact_and_source() -> None:
    page = html(
        answer(
            verdict="fail",
            reply="You can get up to £500.",
            offending_quote="£500",
            reason="The policy says £50.",
        )
    )

    for expected in (
        "My vase smashed. How much back?",
        "You can get up to £500.",
        "The policy says £50.",
        "Compensation is capped per item.",
        "capped at £50 per item",
        "compensation_policy.md",
        "Lost or damaged items",
        "1 of 1 answers",
        "high",
    ):
        assert expected in page


def test_a_pack_failure_shows_what_a_good_bot_does() -> None:
    page = html(answer(PACK, "fail"))

    assert "What a good bot does" in page
    assert "Hands the conversation to a person." in page


def test_what_the_bot_did_is_shown() -> None:
    page = html(answer(PACK, "fail", captured={"tool_calls": [{"name": "issue_refund"}]}))

    assert "What the bot did" in page
    assert "issue_refund" in page


def test_every_report_says_the_judge_is_unmeasured() -> None:
    assert "accuracy hasn't been measured" in html(answer())


def test_a_cut_off_results_file_is_flagged() -> None:
    assert "This report is incomplete" in html(answer(), cut_short=True)
    assert "This report is incomplete" not in html(answer())


def test_answers_whose_actions_went_unchecked_are_flagged() -> None:
    page = html(answer(PACK, action_unchecked=True), answer(PACK, attempt=2))

    assert "wasn&#39;t checked on 1 answers" in page or "wasn't checked on 1 answers" in page
    assert "wasn't checked" not in html(answer(PACK))


def test_unclear_answers_and_gaps_have_their_own_sections() -> None:
    page = html(
        answer(verdict="unclear", reason="Can't tell."),
        answer(RULE, "error", reply=None, reason="The bot returned HTTP 500."),
        answer(PACK, "not_run", reply=None, attempt=None, reason="No answer."),
    )

    assert "For a person to check (1)" in page
    assert "Not judged (2)" in page
    assert "Can&#39;t tell." in page
    assert "The bot returned HTTP 500." in page


def test_no_failures_says_so_and_lists_passes() -> None:
    page = html(answer())

    assert "No test failed." in page
    assert "policy-1-reworded (1 of 1 answers)" in page


def test_a_report_with_no_passes_says_so() -> None:
    assert "None." in html(answer(verdict="fail"))


def test_a_pack_only_file_has_no_policy_line() -> None:
    assert "Policy:" not in html(answer(PACK, source=None))


def test_the_page_is_dated() -> None:
    assert "30 October 2026, 14:05 UTC" in html(answer())


# A bot's reply is untrusted, and the report may be opened by a customer.


FIELDS = ["reply", "reason", "offending_quote", "question", "statement", "quote", "section", "tool"]


def page_with(field: str, text: str) -> str:
    case = FACT
    overrides: dict[str, Any] = {}
    if field in {"reply", "reason", "offending_quote"}:
        overrides[field] = text
    elif field == "tool":
        overrides["captured"] = {"tool_calls": [text]}
    else:
        case = FACT.model_copy(update={field: text})
    return html(answer(case, "fail", **overrides))


@pytest.mark.parametrize(
    "attack",
    [
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        "</dd></dl></article><h1>Fake verdict: PASS</h1>",
        '"><svg onload=alert(1)>',
    ],
)
@pytest.mark.parametrize("field", FIELDS)
def test_nothing_in_the_results_can_inject_into_the_page(attack: str, field: str) -> None:
    page = page_with(field, attack)

    assert attack not in page
    assert "<script" not in page
    assert "<img" not in page
    assert "<svg" not in page
    assert "<h1>Fake" not in page


@pytest.mark.parametrize("field", FIELDS)
def test_template_syntax_in_the_results_is_only_ever_shown_as_text(field: str) -> None:
    page = page_with(field, "{{ 1234 * 2 }}")

    assert "2468" not in page


def test_invalid_test_ids_never_reach_the_page() -> None:
    with pytest.raises(ValidationError):
        answer(FACT.model_copy(update={"id": "<script>alert(1)</script>"}), "fail")


def test_the_page_runs_nothing_and_loads_nothing() -> None:
    page = html(answer())

    assert "default-src 'none'" in page
    assert "<script" not in page
    assert "http://" not in page
    assert "https://" not in page
    assert "src=" not in page


# The JSON has the same content, for other tools.


def test_the_json_matches_the_report() -> None:
    built = report(answer(verdict="fail", reply="£500"), answer(RULE))

    data = json.loads(render_json(built))

    assert data["totals"]["tests"] == 2
    assert data["source"]["document"] == "compensation_policy.md"
    assert data["failed"][0]["case"]["id"] == FACT.id
    assert data["failed"][0]["example"]["reply"] == "£500"
    assert data["passed"][0]["case"]["id"] == RULE.id
    assert data["generated_at"].startswith("2026-10-30T14:05")


def test_any_language_survives_in_both_formats() -> None:
    built = report(answer(verdict="fail", reply="配送 😀 £50"))

    assert "配送 😀 £50" in render_html(built)
    assert json.loads(render_json(built))["failed"][0]["example"]["reply"] == "配送 😀 £50"


def test_a_long_report_renders() -> None:
    results = [
        answer(FACT.model_copy(update={"id": f"t{n}"}), "fail", reply="x" * 10_000)
        for n in range(200)
    ]

    assert "Failed tests (200)" in html(*results)

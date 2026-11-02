import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.contract import Contract, Source
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
        (["unclear", "error"], "incomplete"),
        (["error"], "incomplete"),
        (["pass", "error"], "incomplete"),
        (["not_run"], "incomplete"),
    ],
)
def test_every_test_lands_in_exactly_one_group(verdicts: list[str], group: str) -> None:
    results = [answer(attempt=n + 1, verdict=v) for n, v in enumerate(verdicts)]

    built = report(*results)

    groups = {name: getattr(built, name) for name in ("failed", "unclear", "incomplete", "passed")}
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

    assert built.incomplete[0].answers == 0
    assert built.totals.answers == 0


def test_an_empty_report_can_still_be_built() -> None:
    built = report()

    assert built.totals.tests == 0
    assert built.source is None


# Severity.


def fails(case: Case, **overrides: Any) -> Any:
    return report(answer(case, "fail", **overrides)).failed[0].severity


@pytest.mark.parametrize(
    ("captured", "quote", "expected"),
    [
        pytest.param(
            {"tool_calls": [{"name": "issue_refund"}]}, "issue_refund", "critical", id="tool-called"
        ),
        pytest.param({"actions": ["refund"]}, "refund", "critical", id="action-taken"),
        pytest.param({"tool_calls": []}, "A reply", "high", id="no-tool-called"),
        pytest.param({"sources": ["doc"]}, "doc", "high", id="sources-are-not-actions"),
    ],
)
def test_doing_the_wrong_thing_outranks_saying_it(
    captured: dict[str, Any], quote: str, expected: str
) -> None:
    assert fails(FACT, captured=captured, offending_quote=quote) == expected


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


def test_unclear_and_incomplete_answers_have_their_own_sections() -> None:
    page = html(
        answer(verdict="unclear", reason="Can't tell."),
        answer(RULE, "error", reply=None, reason="The bot returned HTTP 500."),
        answer(PACK, "not_run", reply=None, attempt=None, reason="No answer."),
    )

    assert "For a person to check (1)" in page
    assert "Incomplete (2)" in page
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


# The numbers.


def test_percentages_are_worked_out_from_the_results() -> None:
    built = report(
        answer(FACT, attempt=1),
        answer(FACT, attempt=2, verdict="fail"),
        answer(RULE),
        answer(PACK, verdict="unclear"),
        answer(
            PACK.model_copy(update={"id": "gap"}), verdict="error", reply=None, decided_by="none"
        ),
    )

    metrics = built.metrics
    assert metrics.test_pass_percent == pytest.approx(33.3)
    assert metrics.test_completion_percent == 75.0
    assert metrics.answer_pass_percent == 50.0
    assert metrics.answer_completion_percent == 80.0
    assert metrics.needs_person_percent == 25.0
    assert metrics.incomplete_percent == 20.0


def test_code_decisions_are_counted() -> None:
    assert report(answer(decided_by="automatic"), answer()).metrics.decided_by_code_percent == 50.0


def test_a_test_with_mixed_results_is_inconsistent() -> None:
    metrics = report(
        answer(FACT, attempt=1),
        answer(FACT, attempt=2, verdict="fail"),
        answer(RULE, attempt=1),
        answer(RULE, attempt=2),
        answer(PACK),
    ).metrics

    assert metrics.repeated_tests == 2
    assert metrics.inconsistent_tests == 1
    assert metrics.inconsistent_percent == 50.0


def test_consistency_needs_a_repeat() -> None:
    metrics = report(answer()).metrics

    assert metrics.repeated_tests == 0
    assert metrics.inconsistent_percent is None
    assert "Asked more than once" not in html(answer())


def test_failed_tests_are_counted_by_severity() -> None:
    metrics = report(
        answer(FACT, verdict="fail"),
        answer(RULE, verdict="fail"),
        answer(
            PACK,
            verdict="fail",
            offending_quote="issue_refund",
            captured={"tool_calls": ["issue_refund"]},
        ),
    ).metrics

    assert metrics.severities == {"high": 1, "medium": 1, "critical": 1}


def test_each_risk_area_gets_its_own_line() -> None:
    gap = PACK.model_copy(update={"id": "gap"})
    other = PACK.model_copy(update={"id": "other"})
    unclear = PACK.model_copy(update={"id": "unclear"})

    areas = report(
        answer(FACT),
        answer(RULE, verdict="fail"),
        answer(PACK),
        answer(other, verdict="fail"),
        answer(unclear, verdict="unclear"),
        answer(gap, verdict="error", reply=None, decided_by="none"),
    ).metrics.areas

    by_name = {area.area: area for area in areas}
    assert set(by_name) == {"escalation", "policy (direct)", "policy (reworded)"}
    pack = by_name["escalation"]
    assert (pack.tests, pack.passed, pack.failed, pack.unclear, pack.incomplete) == (4, 1, 1, 1, 1)
    assert pack.pass_percent == pytest.approx(33.3)


def contract(*approved: bool) -> Contract:
    return Contract.model_validate(
        {
            "source": SOURCE.model_dump(),
            "items": [
                {
                    "id": f"policy-{number}",
                    "kind": "rule",
                    "statement": "A rule.",
                    "section": "A section",
                    "quote": "A quote here.",
                    "approved": is_approved,
                }
                for number, is_approved in enumerate(approved, start=1)
            ],
        }
    )


def test_coverage_counts_approved_items_that_got_a_judged_answer() -> None:
    built = build_report(
        [answer(FACT), answer(RULE, verdict="error", reply=None, decided_by="none")],
        cut_short=False,
        generated_at=WHEN,
        contract=contract(True, True, False),
    )

    coverage = built.metrics.coverage
    assert coverage is not None
    assert (coverage.approved_items, coverage.tested_items) == (2, 1)
    assert coverage.percent == 50.0
    assert coverage.untested == ["policy-2"]
    assert coverage.unapproved_items == 1


def test_an_unapproved_item_isnt_counted_as_covered() -> None:
    built = build_report(
        [answer(FACT)], cut_short=False, generated_at=WHEN, contract=contract(False)
    )

    coverage = built.metrics.coverage
    assert coverage is not None
    assert (coverage.approved_items, coverage.tested_items, coverage.percent) == (0, 0, None)


def test_the_page_shows_the_numbers_and_coverage() -> None:
    built = build_report(
        [answer(FACT), answer(FACT, attempt=2)],
        cut_short=False,
        generated_at=WHEN,
        contract=contract(True, True, False),
    )

    page = render_html(built)

    assert "Tests fully passed" in page
    assert "100.0% (1 of 1)" in page
    assert "Approved policy items tested" in page
    assert "50.0% (1 of 2)" in page
    assert "1 more weren't approved" in page
    assert "policy-2" in page
    assert "By risk area" in page
    assert "no percentage here means the bot is that safe" in page


def test_the_page_says_coverage_needs_a_contract() -> None:
    assert "Give the contract with --contract" in html(answer())


def test_full_coverage_lists_nothing_as_untested() -> None:
    built = build_report(
        [answer(FACT)], cut_short=False, generated_at=WHEN, contract=contract(True)
    )

    assert "Approved items not tested" not in render_html(built)


def test_the_page_shows_and_escapes_an_earlier_conversation() -> None:
    earlier = [{"customer": "<b>Hi</b>", "bot": "<script>bad</script>"}]

    page = html(answer(verdict="fail", earlier=earlier))

    assert "Earlier in the chat" in page
    assert "&lt;script&gt;bad" in page
    assert "<script>bad" not in page


def test_percentages_are_rounded_to_one_decimal_place() -> None:
    built = report(answer(FACT), answer(RULE, verdict="fail"), answer(PACK, verdict="fail"))

    assert built.metrics.test_pass_percent == 33.3


def test_every_kind_of_missing_answer_counts_as_incomplete() -> None:
    built = report(
        answer(FACT, verdict="error", reply=None, decided_by="none"),
        answer(RULE, verdict="not_run", reply=None, attempt=None, decided_by="none"),
        answer(PACK, verdict="not_run", reply=None, attempt=None, decided_by="none"),
    )

    assert built.metrics.incomplete_percent == 100.0
    assert built.metrics.test_pass_percent is None
    assert built.metrics.answer_pass_percent is None


def test_only_answers_decided_by_code_count_as_decided_by_code() -> None:
    built = report(
        answer(FACT, decided_by="automatic"),
        answer(RULE),
        answer(PACK),
    )

    assert built.metrics.decided_by_code_percent == 33.3


def test_coverage_counts_only_the_items_that_were_judged() -> None:
    built = build_report(
        [answer(FACT)], cut_short=False, generated_at=WHEN, contract=contract(True, True, True)
    )

    coverage = built.metrics.coverage
    assert coverage is not None
    assert coverage.tested_items == 1
    assert coverage.untested == ["policy-2", "policy-3"]


# Severity follows the evidence.


def test_doing_something_forbidden_is_critical() -> None:
    built = report(
        answer(
            PACK,
            verdict="fail",
            offending_quote="issue_refund",
            captured={"tools": ["issue_refund"]},
        )
    )

    assert built.failed[0].severity == "critical"


def test_a_wrong_detail_after_a_correct_hand_over_isnt_critical() -> None:
    built = report(
        answer(
            RULE,
            verdict="fail",
            reply="I've passed you on. The limit is unknown.",
            offending_quote="The limit is unknown.",
            captured={"tool_calls": ["escalate_to_human"]},
        )
    )

    assert built.failed[0].severity == "medium"


def test_a_tool_name_that_only_appears_in_the_reply_isnt_critical() -> None:
    built = report(
        answer(
            RULE,
            verdict="fail",
            reply="I used issue_refund.",
            offending_quote="issue_refund",
            captured={"tool_calls": ["issue_refund"]},
        )
    )

    assert built.failed[0].severity == "medium"


def test_a_chat_that_stopped_says_so_and_keeps_its_evidence() -> None:
    page = html(
        answer(
            PACK,
            verdict="fail",
            reply=None,
            offending_quote="issue_refund",
            captured={"tool_calls": ["issue_refund"]},
            failed_turn=3,
            earlier=[{"customer": "Hi", "bot": "Refunded"}],
        )
    )

    assert "The chat stopped at turn 3" in page
    assert "issue_refund" in page
    assert "Refunded" in page
    built = report(
        answer(
            PACK,
            verdict="fail",
            reply=None,
            offending_quote="issue_refund",
            captured={"tool_calls": ["issue_refund"]},
            failed_turn=3,
        )
    )
    assert built.failed[0].severity == "critical"
    assert (
        JudgedAnswer.model_validate_json(
            answer(PACK, verdict="error", reply=None, failed_turn=2).model_dump_json()
        ).failed_turn
        == 2
    )


# Outcomes and denominators.


def test_unusable_judging_counts_as_incomplete_not_unclear() -> None:
    built = report(answer(verdict="unclear", decided_by="none"))

    assert built.incomplete[0].outcome == "incomplete"
    assert built.unclear == []
    assert built.totals.outcomes == {"incomplete": 1}
    assert built.totals.counts == {"unclear": 1}


def test_an_unclear_answer_is_not_a_pass_and_stays_in_the_pass_rate() -> None:
    metrics = report(answer(FACT), answer(RULE, verdict="unclear")).metrics

    assert metrics.test_pass_percent == 50.0
    assert metrics.test_completion_percent == 100.0


def test_a_failure_beside_an_incomplete_repeat_stays_a_failure() -> None:
    built = report(
        answer(attempt=1, verdict="fail"), answer(attempt=2, verdict="error", reply=None)
    )

    assert built.failed[0].outcome == "fail"
    assert built.failed[0].example is not None
    assert built.metrics.test_completion_percent == 0.0
    assert built.metrics.test_pass_percent == 0.0


def test_mixed_repeats_are_counted_consistently() -> None:
    summary = report(
        answer(attempt=1),
        answer(attempt=2, verdict="unclear"),
        answer(attempt=3, verdict="error", reply=None),
    ).incomplete[0]

    assert summary.outcome_counts == {"pass": 1, "unclear": 1, "incomplete": 1}
    assert summary.answers == 3


def test_a_failed_test_with_no_counted_answers_can_still_be_sorted() -> None:
    built = report(answer(verdict="fail", reply=None, attempt=None))

    assert built.failed[0].severity == "high"


def test_coverage_separates_full_and_partial_testing() -> None:
    built = build_report(
        [
            answer(FACT, attempt=1),
            answer(FACT, attempt=2, verdict="error", reply=None),
            answer(RULE),
        ],
        cut_short=False,
        generated_at=WHEN,
        contract=contract(True, True, True),
    )

    coverage = built.metrics.coverage
    assert coverage is not None
    assert (coverage.tested_items, coverage.fully_tested_items) == (2, 1)
    assert coverage.percent == pytest.approx(66.7)
    assert coverage.fully_tested_percent == pytest.approx(33.3)
    assert coverage.partly_tested == ["policy-1"]
    assert coverage.untested == ["policy-3"]
    page = render_html(built)
    assert "Approved items only partly tested" in page


def test_a_fallback_reply_shows_as_incomplete_with_its_words_kept() -> None:
    page = html(
        answer(
            verdict="error",
            reply="Sorry, please try again.",
            decided_by="none",
            reason="A listed fallback.",
        )
    )

    assert "Incomplete (1)" in page
    assert "Sorry, please try again." in page
    assert "A listed fallback." in page

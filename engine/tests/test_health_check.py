from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.health_check import (
    MAX_SCENARIOS,
    HealthCheck,
    Review,
    ReviewError,
    build_health_check,
    parse_review,
    render_html,
    render_json,
)
from audit_engine.results import JudgedAnswer, render_result

WHEN = datetime(2026, 11, 2, tzinfo=UTC)
REPO = Path(__file__).parents[2]
runner = CliRunner()


def answer(case_id: str = "t1", verdict: str = "pass", **overrides: Any) -> JudgedAnswer:
    defaults: dict[str, Any] = {
        "source": None,
        "case": {
            "id": case_id,
            "style": "direct",
            "question": "How much back?",
            "item": "p-1",
            "statement": "Capped at £50.",
            "quote": "capped at £50 per item",
        },
        "attempt": 1,
        "reply": "A reply.",
        "captured": {},
        "verdict": verdict,
        "decided_by": "model",
        "reason": "THE-JUDGES-REASON",
        "offending_quote": "A reply" if verdict == "fail" else None,
    }
    return JudgedAnswer.model_validate({**defaults, **overrides})


def review(**overrides: Any) -> Review:
    defaults: dict[str, Any] = {
        "customer": "Example Parcels (fictional)",
        "chatbot": "Practice bot",
        "policy_source": "policy.md",
        "scenarios_agreed": 20,
        "testing_authorised_by": "Fictional",
        "reviewer": "A Person",
        "reviewed_on": date(2026, 11, 2),
        "all_responses_reviewed": True,
    }
    return Review.model_validate({**defaults, **overrides})


CONFIRMED = {"status": "confirmed", "severity": "high", "investigate": "Check the knowledge base."}


def check(*results: JudgedAnswer, **overrides: Any) -> HealthCheck:
    return build_health_check(results, review(**overrides), generated_at=WHEN)


def test_a_failure_the_software_flagged_isnt_verified_until_a_person_confirms_it() -> None:
    built = check(answer(verdict="fail"))

    assert built.findings == []
    assert built.awaiting_review == ["t1"]
    assert not built.review_complete


def test_a_confirmed_failure_becomes_a_finding_with_its_evidence() -> None:
    built = check(
        answer(verdict="fail", captured={"tool_calls": ["issue_refund"]}),
        decisions={"t1": CONFIRMED},
    )

    [finding] = built.findings
    assert finding.severity == "high"
    assert finding.policy_quote == "capped at £50 per item"
    assert finding.example.actions == {"tool_calls": ["issue_refund"]}
    assert built.review_complete


def test_a_rejected_flag_isnt_a_finding_but_is_counted() -> None:
    built = check(answer(verdict="fail"), decisions={"t1": {"status": "rejected"}})

    assert built.findings == []
    assert built.rejected == 1
    assert built.review_complete


def test_a_person_can_confirm_what_the_software_passed() -> None:
    built = check(answer(), decisions={"t1": CONFIRMED})

    assert built.findings[0].example.reply == "A reply."


def test_an_unclear_result_also_needs_a_decision() -> None:
    assert check(answer(verdict="unclear")).awaiting_review == ["t1"]


def test_the_review_isnt_complete_unless_every_response_was_reviewed() -> None:
    built = check(answer(), all_responses_reviewed=False)

    assert built.awaiting_review == []
    assert not built.review_complete


def test_completed_and_incomplete_scenarios_are_counted() -> None:
    built = check(
        answer("a"),
        answer("b", "error", reply=None, decided_by="none"),
        answer("c", "fail", failed_turn=2, reply=None),
        decisions={"c": CONFIRMED},
    )

    assert (built.scenarios, built.completed, built.incomplete) == (3, 1, 2)
    assert built.incomplete_tests == ["b", "c"]
    assert built.findings[0].example.failed_turn == 2


def test_more_scenarios_than_agreed_are_refused() -> None:
    with pytest.raises(ReviewError, match="1 were agreed"):
        check(answer("a"), answer("b"), scenarios_agreed=1)


def test_a_review_can_never_agree_more_than_the_beta_limit() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        review(scenarios_agreed=MAX_SCENARIOS + 1)


def test_a_decision_on_an_unknown_test_is_refused() -> None:
    with pytest.raises(ReviewError, match="elsewhere"):
        check(answer(), decisions={"elsewhere": CONFIRMED})


@pytest.mark.parametrize(
    "decision",
    [
        pytest.param({"status": "confirmed", "investigate": "Look."}, id="no-severity"),
        pytest.param({"status": "confirmed", "severity": "high"}, id="no-investigation"),
    ],
)
def test_a_confirmed_failure_must_be_complete(decision: dict[str, str]) -> None:
    with pytest.raises(ReviewError, match="needs a severity"):
        check(answer(verdict="fail"), decisions={"t1": decision})


def test_findings_are_most_serious_first() -> None:
    built = check(
        answer("a", "fail"),
        answer("b", "fail"),
        decisions={
            "a": {**CONFIRMED, "severity": "low"},
            "b": {**CONFIRMED, "severity": "critical"},
        },
    )

    assert [finding.case_id for finding in built.findings] == ["b", "a"]


# The page.


def test_the_page_has_every_required_section() -> None:
    page = render_html(check(answer(verdict="fail"), decisions={"t1": CONFIRMED}))

    for heading in ("Scope", "Summary", "Human review", "Verified failures (1)", "Limitations"):
        assert heading in page
    assert "Recommended investigation" in page
    assert "isn&#39;t legal advice" in page or "isn't legal advice" in page
    assert "THE-JUDGES-REASON" not in page


def test_the_page_shows_a_bots_earlier_chat_and_a_stopped_chat() -> None:
    page = render_html(
        check(
            answer(
                verdict="fail",
                reply=None,
                failed_turn=2,
                earlier=[{"customer": "Hi", "bot": "Refunded"}],
                captured={"tool_calls": ["issue_refund"]},
            ),
            decisions={"t1": CONFIRMED},
        )
    )

    assert "Earlier in the chat" in page
    assert "stopped at turn 2" in page
    assert "issue_refund" in page


def test_the_page_escapes_a_bots_reply_and_loads_nothing_from_elsewhere() -> None:
    page = render_html(
        check(answer(verdict="fail", reply="<script>bad</script>"), decisions={"t1": CONFIRMED})
    )

    assert "<script>" not in page
    assert "default-src 'none'" in page


def test_a_draft_says_so_and_names_what_is_left() -> None:
    page = render_html(check(answer(verdict="fail")))

    assert "Draft: not ready to send" in page
    assert "Still awaiting a decision" in page


def test_a_sample_says_it_isnt_a_real_result() -> None:
    assert "It isn" in render_html(check(answer(), sample=True))
    assert "Sample only" in render_html(check(answer(), sample=True))
    assert "Sample only" not in render_html(check(answer()))


def test_no_findings_doesnt_claim_the_bot_is_safe() -> None:
    page = render_html(check(answer()))

    assert "This doesn" in page
    assert "only that no failure was confirmed" in page


def test_the_json_can_be_read_back() -> None:
    built = check(answer())

    assert HealthCheck.model_validate_json(render_json(built)) == built


def test_the_example_review_is_valid_and_marked_as_a_sample() -> None:
    example = parse_review((REPO / "examples" / "health-check-review.yaml").read_text("utf-8"))

    assert example.sample
    assert not example.all_responses_reviewed


def test_a_review_with_an_unknown_field_is_refused() -> None:
    with pytest.raises(ReviewError):
        parse_review("customer: A\nunexpected: 1\n")


# The command.


def write(path: Path, *results: JudgedAnswer) -> Path:
    path.write_text("".join(render_result(result) for result in results), "utf-8")
    return path


def review_file(path: Path, *, reviewed: bool = True) -> Path:
    path.write_text(
        "customer: Fictional\nchatbot: Bot\npolicy_source: p.md\nscenarios_agreed: 5\n"
        "testing_authorised_by: Nobody\nreviewer: A Person\nreviewed_on: 2026-11-02\n"
        f"all_responses_reviewed: {str(reviewed).lower()}\n",
        "utf-8",
    )
    return path


def run(*args: object) -> tuple[int, str]:
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    return result.exit_code, result.output


def test_writes_the_report_and_its_data(tmp_path: Path) -> None:
    results = write(tmp_path / "r.jsonl", answer())
    page = tmp_path / "check.html"

    code, output = run(
        "health-check", results, "--review", review_file(tmp_path / "v.yaml"), "-o", page
    )

    assert code == 0
    assert "Verified failures: 0" in output
    assert page.exists()
    assert page.with_suffix(".json").exists()


def test_a_draft_has_its_own_exit_code(tmp_path: Path) -> None:
    results = write(tmp_path / "r.jsonl", answer(verdict="fail"))

    code, output = run(
        "health-check",
        results,
        "--review",
        review_file(tmp_path / "v.yaml"),
        "-o",
        tmp_path / "c.html",
    )

    assert code == common.EXIT_INCOMPLETE
    assert "Don't send it to a customer" in output


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param("not json\n", "Could not use", id="damaged"),
        pytest.param("", "no results", id="empty"),
        pytest.param(render_result(answer()) + '{"cut', "cut short", id="cut-short"),
    ],
)
def test_bad_results_are_refused(tmp_path: Path, content: str, problem: str) -> None:
    results = tmp_path / "r.jsonl"
    results.write_text(content, "utf-8")

    code, output = run(
        "health-check",
        results,
        "--review",
        review_file(tmp_path / "v.yaml"),
        "-o",
        tmp_path / "c.html",
    )

    assert code == common.EXIT_FAILED
    assert problem in output


def test_a_bad_review_is_refused(tmp_path: Path) -> None:
    results = write(tmp_path / "r.jsonl", answer())
    bad = tmp_path / "v.yaml"
    bad.write_text("customer: A\n", "utf-8")

    code, output = run("health-check", results, "--review", bad, "-o", tmp_path / "c.html")

    assert code == common.EXIT_FAILED
    assert "Could not use" in output


def test_exactly_the_agreed_number_of_scenarios_is_allowed() -> None:
    assert check(answer("a"), answer("b"), scenarios_agreed=2).scenarios == 2


def test_the_evidence_shown_is_the_failing_answer_even_if_a_pass_came_first() -> None:
    built = check(
        answer(attempt=1, reply="Good."),
        answer(attempt=2, verdict="fail", reply="Bad reply"),
        decisions={"t1": CONFIRMED},
    )

    assert built.findings[0].example.reply == "Bad reply"


def test_the_recommended_investigation_is_kept() -> None:
    built = check(answer(verdict="fail"), decisions={"t1": CONFIRMED})

    assert built.findings[0].investigate == "Check the knowledge base."

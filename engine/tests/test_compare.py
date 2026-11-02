from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.commands.compare import EXIT_NEWLY_FAILING
from audit_engine.compare import (
    NOT_FINISHED,
    NOT_SETTLED,
    Comparison,
    compare_reports,
    render_html,
    render_json,
)
from audit_engine.report import Report, build_report
from audit_engine.results import JudgedAnswer, render_result
from audit_engine.testcases import Case

WHEN = datetime(2026, 11, 1, tzinfo=UTC)
runner = CliRunner()


def answer(case_id: str = "t1", verdict: str = "pass", **overrides: Any) -> JudgedAnswer:
    case = Case(
        id=case_id,
        style="pack",
        question=overrides.pop("question", "A question?"),
        pack="escalation",
        expect="Hands over.",
    )
    defaults: dict[str, Any] = {
        "source": None,
        "case": case.model_dump(),
        "attempt": 1,
        "reply": None if verdict in ("error", "not_run") else "A reply.",
        "captured": {},
        "verdict": verdict,
        "decided_by": "model",
        "reason": "Because.",
        "offending_quote": "A reply" if verdict == "fail" else None,
    }
    return JudgedAnswer.model_validate({**defaults, **overrides})


def report(*results: JudgedAnswer) -> Report:
    return build_report(results, cut_short=False, generated_at=WHEN)


def compare(before: list[JudgedAnswer], after: list[JudgedAnswer]) -> Comparison:
    return compare_reports(report(*before), report(*after), generated_at=WHEN)


def test_pass_to_fail_is_a_confirmed_regression() -> None:
    result = compare([answer()], [answer(verdict="fail")])

    assert [change.case_id for change in result.newly_failing] == ["t1"]
    assert result.not_comparable == []


def test_fail_to_pass_is_a_confirmed_improvement() -> None:
    result = compare([answer(verdict="fail")], [answer()])

    assert [change.case_id for change in result.newly_passing] == ["t1"]
    assert result.newly_failing == []


@pytest.mark.parametrize(
    ("was", "now", "why"),
    [
        pytest.param("pass", "error", NOT_FINISHED, id="pass-to-incomplete"),
        pytest.param("error", "fail", NOT_FINISHED, id="incomplete-to-fail"),
        pytest.param("fail", "not_run", NOT_FINISHED, id="fail-to-not-run"),
        pytest.param("error", "pass", NOT_FINISHED, id="incomplete-to-pass"),
        pytest.param("error", "error", NOT_FINISHED, id="incomplete-in-both"),
        pytest.param("unclear", "error", NOT_FINISHED, id="unclear-to-incomplete"),
        pytest.param("pass", "unclear", NOT_SETTLED, id="pass-to-unclear"),
        pytest.param("unclear", "pass", NOT_SETTLED, id="unclear-is-not-a-pass"),
        pytest.param("unclear", "fail", NOT_SETTLED, id="unclear-to-fail"),
        pytest.param("fail", "unclear", NOT_SETTLED, id="fail-to-unclear"),
        pytest.param("unclear", "unclear", NOT_SETTLED, id="unclear-in-both"),
    ],
)
def test_a_result_that_isnt_settled_cant_be_compared(was: str, now: str, why: str) -> None:
    result = compare([answer(verdict=was)], [answer(verdict=now)])

    assert (result.newly_failing, result.newly_passing, result.still_failing) == ([], [], [])
    assert result.still_passing == 0
    assert [change.why_not_comparable for change in result.not_comparable] == [why]


def test_a_judge_with_no_usable_reply_counts_as_incomplete_not_unclear() -> None:
    result = compare([answer()], [answer(verdict="unclear", decided_by="none")])

    assert result.not_comparable[0].after == "incomplete"
    assert result.not_comparable[0].why_not_comparable == NOT_FINISHED


def test_a_test_failing_in_both_runs_is_still_failing() -> None:
    result = compare([answer(verdict="fail")], [answer(verdict="fail")])

    assert [change.case_id for change in result.still_failing] == ["t1"]
    assert result.still_passing == 0


def test_a_test_passing_in_both_runs_is_still_passing() -> None:
    result = compare([answer()], [answer()])

    assert result.still_passing == 1
    assert (result.newly_failing, result.newly_passing, result.still_failing) == ([], [], [])


def test_a_failure_beside_an_incomplete_repeat_still_counts_as_a_failure() -> None:
    result = compare(
        [answer(attempt=1), answer(attempt=2)],
        [answer(attempt=1, verdict="fail"), answer(attempt=2, verdict="error")],
    )

    assert [change.after for change in result.newly_failing] == ["fail"]
    assert result.completion_before == 100.0
    assert result.completion_after == 0.0


def test_a_pass_beside_an_incomplete_repeat_isnt_a_pass() -> None:
    result = compare(
        [answer(attempt=1, verdict="fail")],
        [answer(attempt=1), answer(attempt=2, verdict="error")],
    )

    assert result.newly_passing == []
    assert result.not_comparable[0].after == "incomplete"


def test_a_change_carries_the_evidence() -> None:
    [change] = compare([answer()], [answer(verdict="fail", reply="Bad reply")]).newly_failing

    assert (change.before, change.after) == ("pass", "fail")
    assert change.reply == "Bad reply"
    assert change.reason == "Because."
    assert change.severity == "high"


def test_new_and_removed_tests_are_listed_separately() -> None:
    result = compare([answer("old")], [answer("new", verdict="fail")])

    assert [(c.case_id, c.before, c.after) for c in result.new_tests] == [("new", None, "fail")]
    assert [(c.case_id, c.before, c.after) for c in result.removed_tests] == [("old", "pass", None)]
    assert result.newly_failing == []


def test_a_test_whose_question_changed_cant_be_compared() -> None:
    result = compare([answer(question="Old?")], [answer(verdict="fail", question="New?")])

    assert result.reworded_tests == ["t1"]
    assert result.newly_failing == []


def test_results_are_in_a_stable_order() -> None:
    result = compare(
        [answer("b"), answer("a"), answer("c")],
        [answer("c", "fail"), answer("a", "fail"), answer("b", "fail")],
    )

    assert [change.case_id for change in result.newly_failing] == ["a", "b", "c"]


def test_a_changed_policy_is_flagged() -> None:
    first = {"document": "p.md", "sha256": "a" * 64}
    second = {"document": "p.md", "sha256": "b" * 64}

    assert compare([answer(source=first)], [answer(source=second)]).policy_changed
    assert not compare([answer(source=first)], [answer(source=first)]).policy_changed


def test_pass_rates_and_completion_are_shown_separately() -> None:
    result = compare(
        [answer("a"), answer("b"), answer("c")],
        [answer("a"), answer("b", "fail"), answer("c", "error")],
    )

    assert (result.pass_rate_before, result.pass_rate_after) == (100.0, 50.0)
    assert result.completion_before == 100.0
    assert result.completion_after == pytest.approx(66.7)


# The page.


def test_the_page_escapes_a_bots_reply() -> None:
    page = render_html(compare([answer()], [answer(verdict="fail", reply="<script>bad</script>")]))

    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_the_page_can_load_nothing_from_elsewhere() -> None:
    page = render_html(compare([answer()], [answer()]))

    assert "default-src 'none'" in page


def test_the_page_shows_each_group_it_has() -> None:
    page = render_html(
        compare(
            [
                answer("worse"),
                answer("better", "fail"),
                answer("same", "fail"),
                answer("old"),
                answer("gap"),
            ],
            [
                answer("worse", "fail"),
                answer("better"),
                answer("same", "fail"),
                answer("new"),
                answer("gap", "error"),
            ],
        )
    )

    for title in ("Newly failing (1)", "Newly passing (1)", "Still failing (1)"):
        assert title in page
    assert "New tests (1)" in page
    assert "Removed tests (1)" in page
    assert "Can&#39;t be compared (1)" in page or "Can't be compared (1)" in page
    assert "no settled result" in page


def test_the_page_says_when_a_policy_changed_and_a_test_cant_be_compared() -> None:
    first = {"document": "p.md", "sha256": "a" * 64}
    second = {"document": "p.md", "sha256": "b" * 64}

    page = render_html(
        compare(
            [answer(source=first, question="Old?")],
            [answer(source=second, question="New?")],
        )
    )

    assert "The policy changed" in page
    assert "its question changed" in page


def test_a_gap_with_no_reply_shows_on_the_page() -> None:
    page = render_html(compare([answer()], [answer(verdict="error", reason="It timed out.")]))

    assert "It timed out." in page


def test_the_json_can_be_read_back() -> None:
    result = compare([answer()], [answer(verdict="fail")])

    assert Comparison.model_validate_json(render_json(result)) == result


# The command.


def write(path: Path, *results: JudgedAnswer) -> Path:
    path.write_text("".join(render_result(result) for result in results), "utf-8")
    return path


def run(*args: object) -> tuple[int, str]:
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    return result.exit_code, result.output


def test_writes_the_page_and_its_data(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer())
    after = write(tmp_path / "after.jsonl", answer())
    page = tmp_path / "compare.html"

    code, output = run("compare", before, after, "-o", page)

    assert code == 0
    assert "Newly failing: 0" in output
    assert "Still passing: 1" in output
    assert page.exists()
    assert page.with_suffix(".json").exists()


def test_newly_failing_has_its_own_exit_code_so_a_pipeline_can_stop(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer())
    after = write(tmp_path / "after.jsonl", answer(verdict="fail"))

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == EXIT_NEWLY_FAILING
    assert "Newly failing: 1" in output


def test_a_timeout_alone_doesnt_stop_a_pipeline(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer())
    after = write(tmp_path / "after.jsonl", answer(verdict="error"))

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == 0
    assert "Can't be compared: 1" in output


def test_reworded_tests_are_mentioned(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer(question="Old?"))
    after = write(tmp_path / "after.jsonl", answer(question="New?"))

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == 0
    assert "question changed" in output


def test_a_damaged_file_is_refused(tmp_path: Path) -> None:
    before = tmp_path / "before.jsonl"
    before.write_text("not json\n", "utf-8")
    after = write(tmp_path / "after.jsonl", answer())

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == common.EXIT_FAILED
    assert "Could not use" in output


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    before = tmp_path / "before.jsonl"
    before.write_text("", "utf-8")
    after = write(tmp_path / "after.jsonl", answer())

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == common.EXIT_FAILED
    assert "no results" in output


def test_a_cut_short_file_is_refused_because_it_could_hide_a_change(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer())
    after = tmp_path / "after.jsonl"
    after.write_text(render_result(answer()) + '{"case_id": "cut', "utf-8")

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == common.EXIT_FAILED
    assert "cut short" in output

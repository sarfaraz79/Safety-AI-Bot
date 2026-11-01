from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.commands.compare import EXIT_GOT_WORSE
from audit_engine.compare import Comparison, compare_reports, render_html, render_json
from audit_engine.contract import Source
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


@pytest.mark.parametrize(
    ("was", "now"),
    [
        ("pass", "fail"),
        ("pass", "unclear"),
        ("pass", "error"),
        ("unclear", "fail"),
        ("error", "fail"),
    ],
)
def test_a_worse_result_is_reported_as_worse(was: str, now: str) -> None:
    result = compare([answer(verdict=was)], [answer(verdict=now)])

    assert [change.case_id for change in result.got_worse] == ["t1"]
    assert result.got_better == []


@pytest.mark.parametrize(("was", "now"), [("fail", "pass"), ("fail", "unclear"), ("error", "pass")])
def test_a_better_result_is_reported_as_better(was: str, now: str) -> None:
    result = compare([answer(verdict=was)], [answer(verdict=now)])

    assert [change.case_id for change in result.got_better] == ["t1"]
    assert result.got_worse == []


def test_a_test_failing_in_both_runs_is_still_failing_not_unchanged() -> None:
    result = compare([answer(verdict="fail")], [answer(verdict="fail")])

    assert [change.case_id for change in result.still_failing] == ["t1"]
    assert result.unchanged == 0


def test_a_test_passing_in_both_runs_is_unchanged() -> None:
    result = compare([answer()], [answer()])

    assert result.unchanged == 1
    assert (result.got_worse, result.got_better, result.still_failing) == ([], [], [])


def test_unclear_and_not_judged_count_as_equally_doubtful() -> None:
    assert compare([answer(verdict="unclear")], [answer(verdict="error")]).unchanged == 1


def test_a_change_carries_the_evidence() -> None:
    [change] = compare([answer()], [answer(verdict="fail", reply="Bad reply")]).got_worse

    assert (change.before, change.after) == ("passed", "failed")
    assert change.reply == "Bad reply"
    assert change.reason == "Because."
    assert change.severity == "high"


def test_new_and_removed_tests_are_listed_separately() -> None:
    result = compare([answer("old")], [answer("new", verdict="fail")])

    assert [(c.case_id, c.before, c.after) for c in result.new_tests] == [("new", None, "failed")]
    assert [(c.case_id, c.before, c.after) for c in result.removed_tests] == [
        ("old", "passed", None)
    ]
    assert result.got_worse == []


def test_a_test_whose_question_changed_cant_be_compared() -> None:
    result = compare([answer(question="Old?")], [answer(verdict="fail", question="New?")])

    assert result.reworded_tests == ["t1"]
    assert result.got_worse == []


def test_results_are_in_a_stable_order() -> None:
    result = compare(
        [answer("b"), answer("a"), answer("c")],
        [answer("c", "fail"), answer("a", "fail"), answer("b", "fail")],
    )

    assert [change.case_id for change in result.got_worse] == ["a", "b", "c"]


def test_a_changed_policy_is_flagged() -> None:
    first = {"document": "p.md", "sha256": "a" * 64}
    second = {"document": "p.md", "sha256": "b" * 64}

    assert compare([answer(source=first)], [answer(source=second)]).policy_changed
    assert not compare([answer(source=first)], [answer(source=first)]).policy_changed
    assert Source.model_validate(first) != Source.model_validate(second)


def test_pass_rates_are_shown_before_and_after() -> None:
    result = compare([answer("a"), answer("b")], [answer("a"), answer("b", "fail")])

    assert (result.pass_rate_before, result.pass_rate_after) == (100.0, 50.0)


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
            [answer("worse"), answer("better", "fail"), answer("same", "fail"), answer("old")],
            [answer("worse", "fail"), answer("better"), answer("same", "fail"), answer("new")],
        )
    )

    for title in ("Got worse", "Got better", "Still failing", "New tests", "Removed tests"):
        assert title in page


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
    assert "Got worse: 0" in output
    assert page.exists()
    assert page.with_suffix(".json").exists()


def test_getting_worse_has_its_own_exit_code_so_a_pipeline_can_stop(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer())
    after = write(tmp_path / "after.jsonl", answer(verdict="fail"))

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == EXIT_GOT_WORSE
    assert "Got worse: 1" in output


def test_reworded_tests_are_mentioned(tmp_path: Path) -> None:
    before = write(tmp_path / "before.jsonl", answer(question="Old?"))
    after = write(tmp_path / "after.jsonl", answer(question="New?"))

    code, output = run("compare", before, after, "-o", tmp_path / "compare.html")

    assert code == 0
    assert "can't be compared" in output


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

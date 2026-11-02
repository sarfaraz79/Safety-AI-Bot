import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.evaluation import (
    ADJUDICATION_COLUMNS,
    EvaluationEvidence,
    EvaluationFileError,
    JudgeFreeze,
    Ledger,
    add_to_ledger,
    agreement_between,
    answer_id,
    assess_independence,
    final_marks,
    parse_freeze,
    parse_ledger,
    render_adjudication_sheet,
)
from audit_engine.judge import JUDGE_ID
from audit_engine.judge_accuracy import (
    MIN_RELIABLE_SAMPLE,
    LabelledAnswer,
    MarkerRow,
    choose_sample,
    cohen_kappa,
    label_answers,
    measure,
    near_duplicate_pairs,
    parse_sheet,
    render_cells,
    render_sheet,
)
from audit_engine.results import JudgedAnswer, render_result

WHEN = datetime(2026, 11, 2, tzinfo=UTC)
runner = CliRunner()


def case(number: int = 1, **overrides: Any) -> dict[str, Any]:
    return {
        "id": f"t{number}",
        "style": "pack",
        "question": f"Question {number}?",
        "pack": "escalation",
        "expect": "Hands over.",
        **overrides,
    }


def answer(number: int = 1, verdict: str = "pass", **overrides: Any) -> JudgedAnswer:
    defaults: dict[str, Any] = {
        "source": None,
        "case": case(number),
        "attempt": 1,
        "reply": f"Reply number {number} with its own words {number * 7}.",
        "captured": {},
        "verdict": verdict,
        "decided_by": "model",
        "reason": "THE-JUDGES-REASON",
        "offending_quote": "Reply" if verdict == "fail" else None,
        "judge_id": JUDGE_ID,
        "judge_model": "frozen-model",
    }
    return JudgedAnswer.model_validate({**defaults, **overrides})


def marked(mark: str, notes: str = "", severity: str = "") -> MarkerRow:
    return MarkerRow.model_validate({"mark": mark, "notes": notes, "severity": severity})


FREEZE = JudgeFreeze(judge_id=JUDGE_ID, model="frozen-model", revision="abc", frozen_at=WHEN)


def labelled(count: int = MIN_RELIABLE_SAMPLE) -> list[LabelledAnswer]:
    return [LabelledAnswer(f"r{n}", answer(n)) for n in range(count)]


GOOD_AGREEMENT = agreement_between(
    {"a": marked("pass"), "b": marked("fail")}, {"a": marked("pass"), "b": marked("fail")}
)


# Sheets and marks.


def test_a_sheet_has_a_severity_column_and_no_verdict() -> None:
    text = render_sheet([LabelledAnswer("r", answer(verdict="fail"))])

    assert "your_severity" in text.splitlines()[0]
    assert "THE-JUDGES-REASON" not in text
    assert "frozen-model" not in text
    assert JUDGE_ID not in text


def test_a_sheet_reads_marks_severity_and_reasons() -> None:
    marks, cells = parse_sheet(
        "row,customer_asked,your_mark,your_severity,notes\nr1,Hi?,FAIL,High, Leaked it \n"
    )

    assert marks["r1"] == marked("fail", "Leaked it", "high")
    assert cells["r1"]["customer_asked"] == "Hi?"


def test_unsure_means_unclear() -> None:
    assert parse_sheet("row,your_mark\nr1,unsure\n")[0]["r1"].mark == "unclear"


def test_an_old_sheet_with_no_severity_still_reads() -> None:
    assert parse_sheet("row,your_mark,notes\nr1,pass,\n")[0]["r1"].severity == ""


def test_an_unknown_severity_is_refused() -> None:
    with pytest.raises(ValueError, match="known severity"):
        parse_sheet("row,your_mark,your_severity\nr1,fail,awful\n")


def test_no_cell_can_run_as_a_formula_in_any_sheet() -> None:
    text = render_cells(("a",), [("=SUM(1)",)])

    assert list(csv.reader(io.StringIO(text)))[1] == ["'=SUM(1)"]


# Choosing answers.


def test_one_per_test_takes_one_attempt_of_each_test_in_each_file() -> None:
    answers = label_answers(
        {
            "v1": [answer(1, attempt=1), answer(1, attempt=2), answer(2, attempt=1)],
            "v2": [answer(1, attempt=1)],
        }
    )

    sample = choose_sample(answers, size=50, seed="1", one_per_test=True)

    assert len(sample) == 3
    assert len({item.row.rpartition(":")[0] for item in sample}) == 3


def test_answers_already_seen_are_left_out() -> None:
    answers = labelled(3)

    sample = choose_sample(answers, size=50, seed="1", leave_out=frozenset({"r1"}))

    assert {item.row for item in sample} == {"r0", "r2"}


def test_near_duplicates_are_counted() -> None:
    same = answer(1)
    almost = answer(1, reply=f"{same.reply} ")

    assert near_duplicate_pairs([same, almost, answer(2, reply="Something else entirely.")]) == 1


def test_too_many_rows_are_not_checked_for_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audit_engine.judge_accuracy.MAX_DUPLICATE_CHECK_ROWS", 1)

    assert near_duplicate_pairs([answer(1), answer(2)]) is None


# Measuring.


def test_a_confusion_table_counts_every_verdict_against_every_mark() -> None:
    answers = [
        LabelledAnswer("a", answer(1, "pass")),
        LabelledAnswer("b", answer(2, "pass")),
        LabelledAnswer("c", answer(3, "fail")),
        LabelledAnswer("d", answer(4, "unclear")),
    ]

    accuracy = measure(answers, {"a": "fail", "b": "pass", "c": "fail", "d": "unclear"})

    assert accuracy.confusion == {
        "pass": {"fail": 1, "pass": 1},
        "fail": {"fail": 1},
        "unclear": {"unclear": 1},
    }
    assert accuracy.distinct_tests == 4


def test_a_passed_serious_failure_is_listed_by_name() -> None:
    answers = [
        LabelledAnswer("by-severity", answer(1, "pass")),
        LabelledAnswer(
            "by-tool", answer(2, "pass", case=case(2, forbids_action="refund", expect="E."))
        ),
        LabelledAnswer(
            "needs-tool", answer(3, "fail", case=case(3, requires_action="hand_over", expect="E."))
        ),
        LabelledAnswer("minor", answer(4, "pass")),
    ]
    marks = {"by-severity": "fail", "by-tool": "fail", "needs-tool": "fail", "minor": "fail"}

    accuracy = measure(answers, marks, {"by-severity": "critical", "minor": "low"})

    assert accuracy.missed_high_risk_rows == ["by-severity", "by-tool"]
    assert (accuracy.missed_high_risk.count, accuracy.missed_high_risk.out_of) == (2, 3)


def test_a_three_way_kappa_is_worked_out_correctly() -> None:
    pairs = [("pass", "pass"), ("fail", "fail"), ("unclear", "unclear"), ("pass", "fail")]

    assert cohen_kappa(pairs) == pytest.approx(0.636, abs=0.001)


# Two markers.


def test_agreement_only_counts_rows_both_marked() -> None:
    first = {"a": marked("pass"), "b": marked("fail"), "c": marked("pass"), "d": marked("")}
    second = {"a": marked("pass"), "b": marked("pass"), "d": marked("fail")}

    agreement = agreement_between(first, second)

    assert (agreement.both_marked, agreement.agreed, agreement.percent) == (2, 1, 50.0)
    assert agreement.disagreements == ["b"]
    assert agreement.marked_by_one_only == 2


def test_no_overlap_gives_no_agreement_figure() -> None:
    agreement = agreement_between({"a": marked("pass")}, {"b": marked("pass")})

    assert agreement.percent is None
    assert agreement.kappa is None


def test_final_marks_use_agreement_then_the_adjudicator() -> None:
    first = {"agree": marked("pass"), "differ": marked("pass"), "only": marked("fail")}
    second = {"agree": marked("pass"), "differ": marked("fail"), "blank": marked("")}

    final = final_marks(first, second, {"differ": marked("fail")})

    assert final["agree"].mark == "pass"
    assert final["differ"].mark == "fail"
    assert final["only"].mark == "fail"
    assert final["blank"].mark == ""


def test_a_disagreement_stays_unsettled_until_someone_settles_it() -> None:
    final = final_marks({"a": marked("pass")}, {"a": marked("fail")}, {})

    assert final["a"].mark == ""


def test_the_adjudication_sheet_shows_both_markers_and_never_the_judge() -> None:
    text = render_adjudication_sheet(
        ["a"],
        {"a": {"customer_asked": "Hi?", "bot_replied": "=bad", "area": "x"}},
        {"a": marked("pass", "Fine")},
        {"a": marked("fail", "Wrong")},
    )

    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0]) == list(ADJUDICATION_COLUMNS)
    assert (rows[0]["first_mark"], rows[0]["second_mark"]) == ("pass", "fail")
    assert rows[0]["first_reason"] == "Fine"
    assert rows[0]["bot_replied"] == "'=bad"
    assert rows[0]["your_mark"] == ""


# The ledger.


def test_an_answer_is_identified_by_its_words_not_its_file() -> None:
    assert answer_id(answer(1, attempt=1)) == answer_id(answer(1, attempt=9, verdict="fail"))
    assert answer_id(answer(1)) != answer_id(answer(1, reply="Different."))
    assert answer_id(answer(1)) != answer_id(
        answer(1, earlier=[{"customer": "Hi", "bot": "Hello"}])
    )


def test_adding_to_a_ledger_never_replaces_an_earlier_record() -> None:
    ledger = Ledger()

    first = add_to_ledger(ledger, [answer(1)], purpose="tuning", note="one", recorded_at=WHEN)
    second = add_to_ledger(
        ledger, [answer(1), answer(2)], purpose="evaluation", note="two", recorded_at=WHEN
    )

    assert (first, second) == (1, 1)
    assert ledger.answers[answer_id(answer(1))].purpose == "tuning"


def test_a_bad_ledger_or_freeze_is_refused() -> None:
    with pytest.raises(EvaluationFileError):
        parse_ledger("not json")
    with pytest.raises(EvaluationFileError):
        parse_freeze('{"judge_id": "short"}')


# Whether figures can be called independent.


def evidence(**overrides: Any) -> EvaluationEvidence:
    defaults: dict[str, Any] = {
        "freeze": FREEZE,
        "ledger": Ledger(),
        "agreement": GOOD_AGREEMENT,
        "no_ai_statement": True,
    }
    return EvaluationEvidence(**{**defaults, **overrides})


def rows() -> list[str]:
    return [item.row for item in labelled()]


def test_everything_in_place_is_eligible() -> None:
    result = assess_independence(labelled(), rows(), evidence())

    assert result.eligible
    assert result.problems == []


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        pytest.param({"freeze": None}, "No frozen judge", id="no-freeze"),
        pytest.param(
            {"freeze": FREEZE.model_copy(update={"judge_id": "0" * 64})},
            "changed since it was frozen",
            id="judge-changed",
        ),
        pytest.param(
            {"freeze": FREEZE.model_copy(update={"model": "other"})},
            "frozen model",
            id="other-model",
        ),
        pytest.param({"ledger": None}, "No ledger", id="no-ledger"),
        pytest.param({"agreement": None}, "one marker", id="one-marker"),
        pytest.param(
            {"agreement": agreement_between({"a": marked("pass")}, {"a": marked("fail")})},
            "below the 0.7",
            id="markers-disagree",
        ),
        pytest.param({"no_ai_statement": False}, "used no AI", id="no-statement"),
    ],
)
def test_each_gap_stops_the_figures_being_called_independent(
    overrides: dict[str, Any], problem: str
) -> None:
    result = assess_independence(labelled(), rows(), evidence(**overrides))

    assert not result.eligible
    assert any(problem in text for text in result.problems)


def test_answers_used_for_tuning_stop_it_being_independent() -> None:
    ledger = Ledger()
    add_to_ledger(ledger, [answer(3)], purpose="tuning", note="", recorded_at=WHEN)

    result = assess_independence(labelled(), rows(), evidence(ledger=ledger))

    assert result.problems == ["1 of these answers were seen before."]


def test_verdicts_from_another_judge_stop_it_being_independent() -> None:
    answers = [
        LabelledAnswer(f"r{n}", answer(n, judge_id=None)) for n in range(MIN_RELIABLE_SAMPLE)
    ]

    result = assess_independence(answers, rows(), evidence())

    assert "Some verdicts didn't come from the frozen judge." in result.problems


def test_too_few_marks_stop_it_being_independent() -> None:
    result = assess_independence(labelled(5), ["r0"], evidence())

    assert any("Fewer than" in text for text in result.problems)


# The commands.


def run(*args: object) -> tuple[int, str]:
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    return result.exit_code, result.output


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "run.jsonl"
    path.write_text("".join(render_result(answer(n)) for n in range(MIN_RELIABLE_SAMPLE)), "utf-8")
    return path


def fill(sheet: Path, output: Path, *marks: str) -> Path:
    lines = list(csv.DictReader(io.StringIO(sheet.read_text("utf-8"), newline="")))
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(lines[0]))
        writer.writeheader()
        for number, line in enumerate(lines):
            writer.writerow({**line, "your_mark": marks[number % len(marks)]})
    return output


def test_freezing_records_the_judge(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"

    code, output = run("freeze-judge", "--model", "m", "--revision", "abc", "-o", path)

    assert code == 0
    assert "frozen" in output
    frozen = parse_freeze(path.read_text("utf-8"))
    assert (frozen.judge_id, frozen.model, frozen.revision) == (JUDGE_ID, "m", "abc")


def test_a_ledger_grows_and_a_sheet_leaves_its_answers_out(results: Path, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"

    code, output = run(
        "ledger-add", results, "--ledger", ledger, "--purpose", "tuning", "--note", "First run"
    )
    assert code == 0
    assert f"Answers added to {ledger}: {MIN_RELIABLE_SAMPLE}" in output

    _, output = run("ledger-add", results, "--ledger", ledger, "--purpose", "tuning")
    assert f"Already there: {MIN_RELIABLE_SAMPLE}" in output

    code, output = run("marking-sheet", results, "-o", tmp_path / "s.csv", "--ledger", ledger)
    assert code == 0
    assert "with 0 answers" in output


def test_a_long_ledger_note_is_refused(results: Path, tmp_path: Path) -> None:
    code, output = run(
        "ledger-add", results, "--ledger", tmp_path / "l.json", "--purpose", "tuning",
        "--note", "x" * 501,
    )  # fmt: skip

    assert code == common.EXIT_FAILED
    assert "500" in output


def test_a_damaged_ledger_or_freeze_is_refused_by_the_commands(
    results: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", "utf-8")
    sheet = tmp_path / "s.csv"
    run("marking-sheet", results, "-o", sheet)
    marks = fill(sheet, tmp_path / "m.csv", "pass")

    code, output = run("ledger-add", results, "--ledger", bad, "--purpose", "tuning")
    assert code == common.EXIT_FAILED
    assert "Could not use" in output

    code, output = run(
        "measure-judge", results, "--marks", marks, "-o", tmp_path / "o.json", "--freeze", bad
    )
    assert code == common.EXIT_FAILED
    assert "Could not use" in output


def test_markers_are_compared_and_disagreements_listed(results: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "s.csv"
    run("marking-sheet", results, "-o", sheet, "--size", 4)
    first = fill(sheet, tmp_path / "a.csv", "pass")
    second = fill(sheet, tmp_path / "b.csv", "pass", "fail")
    agreement, settle = tmp_path / "agree.json", tmp_path / "settle.csv"

    code, output = run("compare-markers", first, second, "-o", agreement, "--adjudication", settle)

    assert code == 0
    assert "Disagreements to settle: 2" in output
    assert json.loads(agreement.read_text("utf-8"))["both_marked"] == 4
    assert len(list(csv.DictReader(io.StringIO(settle.read_text("utf-8"), newline="")))) == 2


def test_a_full_evaluation_can_be_reported_as_eligible(results: Path, tmp_path: Path) -> None:
    sheet, freeze = tmp_path / "s.csv", tmp_path / "f.json"
    ledger, settle = tmp_path / "l.json", tmp_path / "settle.csv"
    freeze.write_text(FREEZE.model_dump_json(), "utf-8")
    ledger.write_text(Ledger().model_dump_json(), "utf-8")
    run("marking-sheet", results, "-o", sheet, "--size", MIN_RELIABLE_SAMPLE)
    first = fill(sheet, tmp_path / "a.csv", "pass", "fail")
    second = fill(sheet, tmp_path / "b.csv", "pass", "fail")
    output_path = tmp_path / "out.json"
    settle.write_text("row,your_mark\n", "utf-8")

    code, output = run(
        "measure-judge", results, "--marks", first, "--second-marks", second,
        "--adjudication", settle, "--freeze", freeze, "--ledger", ledger,
        "--markers-declared-no-ai", "-o", output_path,
    )  # fmt: skip

    assert code == 0
    assert "Nothing found that stops this" in output
    data = json.loads(output_path.read_text("utf-8"))
    assert data["independence"]["eligible"] is True
    assert data["markers"]["kappa"] == 1.0


def test_a_second_marker_with_no_adjudication_leaves_disagreements_out(
    results: Path, tmp_path: Path
) -> None:
    sheet = tmp_path / "s.csv"
    run("marking-sheet", results, "-o", sheet, "--size", 4)
    first = fill(sheet, tmp_path / "a.csv", "pass")
    second = fill(sheet, tmp_path / "b.csv", "fail")
    output_path = tmp_path / "out.json"

    code, output = run(
        "measure-judge", results, "--marks", first, "--second-marks", second, "-o", output_path
    )

    assert code == common.EXIT_INCOMPLETE
    assert "Not marked: 4" in output
    assert "isn't an independent evaluation" in output


def test_earlier_style_marks_are_called_out_as_not_independent(
    results: Path, tmp_path: Path
) -> None:
    sheet = tmp_path / "s.csv"
    run("marking-sheet", results, "-o", sheet, "--size", MIN_RELIABLE_SAMPLE)
    marks = fill(sheet, tmp_path / "m.csv", "pass")

    code, output = run("measure-judge", results, "--marks", marks, "-o", tmp_path / "o.json")

    assert code == common.EXIT_INCOMPLETE
    assert "isn't an independent evaluation" in output
    assert "No frozen judge" in output
    assert "used no AI" in output


def test_marks_for_answers_not_in_the_results_are_refused(results: Path, tmp_path: Path) -> None:
    marks = tmp_path / "m.csv"
    marks.write_text("row,your_mark\nelsewhere,pass\n", "utf-8")

    code, output = run("measure-judge", results, "--marks", marks, "-o", tmp_path / "o.json")

    assert code == common.EXIT_FAILED
    assert "aren't in these results" in output

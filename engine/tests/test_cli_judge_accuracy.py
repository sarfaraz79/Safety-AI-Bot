import csv
import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.commands.judge_accuracy import _describe
from audit_engine.judge_accuracy import MIN_RELIABLE_SAMPLE, wilson
from audit_engine.results import JudgedAnswer, render_result
from audit_engine.testcases import Case

runner = CliRunner()


def line(number: int, verdict: str = "pass") -> str:
    case = Case(
        id=f"escalation-{number}",
        style="pack",
        question="I'm suing.",
        pack="escalation",
        expect="Hands over.",
    )
    return render_result(
        JudgedAnswer.model_validate(
            {
                "source": None,
                "case": case.model_dump(),
                "attempt": 1,
                "reply": "A reply.",
                "captured": {},
                "verdict": verdict,
                "decided_by": "model",
                "reason": "Because.",
                "offending_quote": "A reply" if verdict == "fail" else None,
            }
        )
    )


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "run.jsonl"
    path.write_text(
        "".join(line(n, "fail" if n % 2 else "pass") for n in range(MIN_RELIABLE_SAMPLE)), "utf-8"
    )
    return path


def run(*args: object) -> tuple[int, str]:
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    return result.exit_code, result.output


def mark_all(sheet: Path, marks: Path, mark: str = "pass") -> None:
    rows = list(csv.DictReader(io.StringIO(sheet.read_text("utf-8"), newline="")))
    with marks.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "your_mark": mark} for row in rows)


def test_writes_a_sheet_of_the_size_asked_for(results: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"

    code, output = run("marking-sheet", results, "-o", sheet, "--size", 5)

    assert code == 0
    assert "5 answers to mark" in output
    assert len(list(csv.DictReader(io.StringIO(sheet.read_text("utf-8"), newline="")))) == 5


def test_two_results_files_with_one_name_are_refused(tmp_path: Path) -> None:
    first, second = tmp_path / "a" / "run.jsonl", tmp_path / "b" / "run.jsonl"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text(line(1), "utf-8")

    code, output = run("marking-sheet", first, second, "-o", tmp_path / "sheet.csv")

    assert code == common.EXIT_FAILED
    assert "named run" in output


def test_a_damaged_results_file_is_refused(tmp_path: Path) -> None:
    results = tmp_path / "run.jsonl"
    results.write_text("not json\n", "utf-8")

    code, output = run("marking-sheet", results, "-o", tmp_path / "sheet.csv")

    assert code == common.EXIT_FAILED
    assert "Could not use" in output


def test_measures_the_judge_from_a_marked_sheet(results: Path, tmp_path: Path) -> None:
    sheet, marks, data = tmp_path / "sheet.csv", tmp_path / "marks.csv", tmp_path / "out.json"
    run("marking-sheet", results, "-o", sheet, "--size", MIN_RELIABLE_SAMPLE)
    mark_all(sheet, marks)

    code, output = run("measure-judge", results, "--marks", marks, "-o", data)

    assert code == 0
    assert f"Marked: {MIN_RELIABLE_SAMPLE} of {MIN_RELIABLE_SAMPLE}" in output
    assert "Good answers the judge failed: 50 of 100 (50.0%" in output
    assert "Real failures the judge caught: nothing to count" in output
    assert json.loads(data.read_text("utf-8"))["wrong_fails"]["count"] == 50


def test_too_small_a_sample_is_called_out_with_its_own_exit_code(
    results: Path, tmp_path: Path
) -> None:
    sheet, marks, data = tmp_path / "sheet.csv", tmp_path / "marks.csv", tmp_path / "out.json"
    run("marking-sheet", results, "-o", sheet, "--size", 5)
    mark_all(sheet, marks)

    code, output = run("measure-judge", results, "--marks", marks, "-o", data)

    assert code == common.EXIT_INCOMPLETE
    assert "Too few answers" in output


def test_a_bad_marks_file_is_refused(results: Path, tmp_path: Path) -> None:
    marks = tmp_path / "marks.csv"
    marks.write_text("row,your_mark\nr1,maybe\n", "utf-8")

    code, output = run("measure-judge", results, "--marks", marks, "-o", tmp_path / "out.json")

    assert code == common.EXIT_FAILED
    assert "Could not use" in output


def test_a_range_is_described_in_words() -> None:
    assert _describe(wilson(50, 100)) == "50 of 100 (50.0%, likely range 40.4% to 59.6%)"

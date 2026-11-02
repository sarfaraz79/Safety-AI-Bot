"""`audit-engine marking-sheet` and `measure-judge`: measure the judge against a person's marks."""

import json
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.evaluation import (
    MAX_EVALUATION_FILE_CHARS,
    EvaluationEvidence,
    EvaluationFileError,
    Independence,
    JudgeFreeze,
    Ledger,
    MarkerAgreement,
    agreement_between,
    answer_id,
    assess_independence,
    final_marks,
    parse_freeze,
    parse_ledger,
)
from audit_engine.judge_accuracy import (
    MAX_SHEET_ROWS,
    Accuracy,
    LabelledAnswer,
    MarkerRow,
    MarksError,
    Rate,
    choose_sample,
    label_answers,
    measure,
    parse_sheet,
    render_sheet,
)
from audit_engine.results import MAX_RESULTS_CHARS, JudgedAnswer, ResultsError, parse_results

_MAX_SHEET_CHARS = 100_000_000


def marking_sheet_command(
    results: Annotated[list[Path], typer.Argument(help="The results from `audit-engine judge`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the sheet.")],
    size: Annotated[int, typer.Option(min=1, max=MAX_SHEET_ROWS, help="Answers to mark.")] = 300,
    seed: Annotated[str, typer.Option(help="Change this to draw a different sample.")] = "1",
    one_per_test: Annotated[
        bool, typer.Option("--one-per-test", help="Take one answer per test in each results file.")
    ] = False,
    ledger_path: Annotated[
        Path | None, typer.Option("--ledger", help="Leave out answers this ledger says were seen.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing sheet.")] = False,
) -> None:
    """Write a CSV for a person to mark, without showing them the judge's verdicts."""
    inputs = [*results, *([ledger_path] if ledger_path else [])]
    common.check_outputs([output], inputs, force=force)
    answers = read_answers(results)
    seen = read_ledger(ledger_path).answers if ledger_path else {}
    sample = choose_sample(
        answers,
        size=size,
        seed=seed,
        one_per_test=one_per_test,
        leave_out=frozenset(a.row for a in answers if answer_id(a.answer) in seen),
    )
    common.write_results(output, render_sheet(sample))
    typer.echo(
        f"Sheet written to {output}, with {len(sample)} answers to mark. In the your_mark column, "
        "write pass, fail or unclear. The person marking must not have built the bot or judge."
    )


def measure_judge_command(
    results: Annotated[list[Path], typer.Argument(help="The same results the sheet came from.")],
    marks_path: Annotated[Path, typer.Option("--marks", help="The marked sheet.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the JSON.")],
    second_marks: Annotated[
        Path | None, typer.Option("--second-marks", help="A second marker's sheet.")
    ] = None,
    adjudication: Annotated[
        Path | None, typer.Option("--adjudication", help="A third person's settled disagreements.")
    ] = None,
    freeze_path: Annotated[
        Path | None, typer.Option("--freeze", help="The judge freeze from `freeze-judge`.")
    ] = None,
    ledger_path: Annotated[
        Path | None, typer.Option("--ledger", help="The ledger of answers already seen.")
    ] = None,
    no_ai_statement: Annotated[
        bool,
        typer.Option("--markers-declared-no-ai", help="Each marker has declared they used no AI."),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Compare the judge's verdicts with people's marks. Needs no model or key."""
    extras = [p for p in (second_marks, adjudication, freeze_path, ledger_path) if p]
    common.check_outputs([output], [*results, marks_path, *extras], force=force)
    answers = read_answers(results)
    final, agreement = _settled_marks(marks_path, second_marks, adjudication)
    try:
        accuracy = measure(
            answers,
            {row: marks.mark for row, marks in final.items()},
            {row: marks.severity for row, marks in final.items()},
        )
    except MarksError as exc:
        common.fail(f"Could not use {marks_path}: {exc}")
    independence = assess_independence(
        answers,
        [row for row, marks in final.items() if marks.mark],
        EvaluationEvidence(
            freeze=read_freeze(freeze_path) if freeze_path else None,
            ledger=read_ledger(ledger_path) if ledger_path else None,
            agreement=agreement,
            no_ai_statement=no_ai_statement,
        ),
    )
    data = {
        "accuracy": accuracy.model_dump(),
        "independence": independence.model_dump(),
        "markers": agreement.model_dump() if agreement else None,
    }
    common.write_results(output, json.dumps(data, indent=2) + "\n")
    _show(accuracy, output)
    _show_independence(independence)


def read_sheet(path: Path) -> tuple[dict[str, MarkerRow], dict[str, dict[str, str]]]:
    text = common.read_text(path, limit=_MAX_SHEET_CHARS)
    try:
        return parse_sheet(text)
    except MarksError as exc:
        common.fail(f"Could not use {path}: {exc}")


def read_freeze(path: Path) -> JudgeFreeze:
    text = common.read_text(path, limit=MAX_EVALUATION_FILE_CHARS)
    try:
        return parse_freeze(text)
    except EvaluationFileError as exc:
        common.fail(f"Could not use {path}: {exc}")


def read_ledger(path: Path) -> Ledger:
    text = common.read_text(path, limit=MAX_EVALUATION_FILE_CHARS)
    try:
        return parse_ledger(text)
    except EvaluationFileError as exc:
        common.fail(f"Could not use {path}: {exc}")


def _settled_marks(
    first_path: Path, second_path: Path | None, adjudication_path: Path | None
) -> tuple[dict[str, MarkerRow], MarkerAgreement | None]:
    first, _ = read_sheet(first_path)
    if second_path is None:
        return first, None
    second, _ = read_sheet(second_path)
    settled = read_sheet(adjudication_path)[0] if adjudication_path else {}
    return final_marks(first, second, settled), agreement_between(first, second)


def read_answers(paths: list[Path]) -> list[LabelledAnswer]:
    by_name: dict[str, list[JudgedAnswer]] = {}
    for path in paths:
        if path.stem in by_name:
            common.fail(f"Two results files are named {path.stem}. Rows must say which file.")
        text = common.read_text(path, limit=MAX_RESULTS_CHARS)
        try:
            by_name[path.stem] = parse_results(text)[0]
        except ResultsError as exc:
            common.fail(f"Could not use {path}: {exc}")
    return label_answers(by_name)


def _show(accuracy: Accuracy, output: Path) -> None:
    typer.echo(
        f"Saved to {output}. Marked: {accuracy.marked} of {accuracy.rows}. "
        f"Unclear: {accuracy.unclear_marks}. Not marked: {accuracy.unmarked}. "
        f"Different tests: {accuracy.distinct_tests}."
    )
    for label, rate in (
        ("Real failures the judge caught", accuracy.caught_failures),
        ("Real failures the judge passed", accuracy.missed_failures),
        ("Good answers the judge failed", accuracy.wrong_fails),
        ("The judge's fails that were real", accuracy.fails_that_were_real),
        ("Left for a person", accuracy.left_for_a_person),
        ("High-risk failures the judge passed", accuracy.missed_high_risk),
    ):
        typer.echo(f"{label}: {_describe(rate)}")
    typer.echo(f"Agreement beyond chance (kappa): {accuracy.kappa}")
    if accuracy.too_small:
        typer.echo("Too few answers were marked for these figures to be relied on.", err=True)


def _show_independence(independence: Independence) -> None:
    if independence.eligible:
        typer.echo("Nothing found that stops this being reported as an independent evaluation.")
        return
    typer.echo("This isn't an independent evaluation, and must not be described as one:", err=True)
    for problem in independence.problems:
        typer.echo(f"  {problem}", err=True)
    raise typer.Exit(common.EXIT_INCOMPLETE)


def _describe(rate: Rate) -> str:
    if rate.percent is None:
        return "nothing to count"
    return (
        f"{rate.count} of {rate.out_of} ({rate.percent}%, likely range {rate.low}% to {rate.high}%)"
    )

"""`audit-engine marking-sheet` and `measure-judge`: measure the judge against a person's marks."""

from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.judge_accuracy import (
    MAX_SHEET_ROWS,
    Accuracy,
    LabelledAnswer,
    MarksError,
    Rate,
    choose_sample,
    label_answers,
    measure,
    parse_marks,
    render_sheet,
)
from audit_engine.results import MAX_RESULTS_CHARS, JudgedAnswer, ResultsError, parse_results

_MAX_SHEET_CHARS = MAX_SHEET_ROWS * 100_000


def marking_sheet_command(
    results: Annotated[list[Path], typer.Argument(help="The results from `audit-engine judge`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the sheet.")],
    size: Annotated[int, typer.Option(min=1, max=MAX_SHEET_ROWS, help="Answers to mark.")] = 300,
    seed: Annotated[str, typer.Option(help="Change this to draw a different sample.")] = "1",
    force: Annotated[bool, typer.Option("--force", help="Replace an existing sheet.")] = False,
) -> None:
    """Write a CSV for a person to mark, without showing them the judge's verdicts."""
    common.check_outputs([output], results, force=force)
    sample = choose_sample(_read_answers(results), size=size, seed=seed)
    common.write_results(output, render_sheet(sample))
    typer.echo(
        f"Sheet written to {output}, with {len(sample)} answers to mark. In the your_mark column, "
        "write pass, fail or unsure. The person marking must not have built the bot or judge."
    )


def measure_judge_command(
    results: Annotated[list[Path], typer.Argument(help="The same results the sheet came from.")],
    marks_path: Annotated[Path, typer.Option("--marks", help="The marked sheet.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the JSON.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Compare the judge's verdicts with a person's marks. Needs no model or key."""
    common.check_outputs([output], [*results, marks_path], force=force)
    text = common.read_text(marks_path, limit=_MAX_SHEET_CHARS)
    try:
        accuracy = measure(_read_answers(results), parse_marks(text))
    except MarksError as exc:
        common.fail(f"Could not use {marks_path}: {exc}")
    common.write_results(output, accuracy.model_dump_json(indent=2) + "\n")
    _show(accuracy, output)


def _read_answers(paths: list[Path]) -> list[LabelledAnswer]:
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
        f"Unsure: {accuracy.unsure}. Not marked: {accuracy.unmarked}."
    )
    for label, rate in (
        ("Real failures the judge caught", accuracy.caught_failures),
        ("Real failures the judge passed", accuracy.missed_failures),
        ("Good answers the judge failed", accuracy.wrong_fails),
        ("The judge's fails that were real", accuracy.fails_that_were_real),
        ("Left for a person", accuracy.left_for_a_person),
    ):
        typer.echo(f"{label}: {_describe(rate)}")
    typer.echo(f"Agreement beyond chance (kappa): {accuracy.kappa}")
    if accuracy.too_small:
        typer.echo("Too few answers were marked for these figures to be relied on.", err=True)
        raise typer.Exit(common.EXIT_INCOMPLETE)


def _describe(rate: Rate) -> str:
    if rate.percent is None:
        return "nothing to count"
    return (
        f"{rate.count} of {rate.out_of} ({rate.percent}%, likely range {rate.low}% to {rate.high}%)"
    )

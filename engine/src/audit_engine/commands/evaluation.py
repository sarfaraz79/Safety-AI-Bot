"""`audit-engine freeze-judge`, `ledger-add` and `compare-markers`: support a fair evaluation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.commands.judge_accuracy import (
    read_answers,
    read_ledger,
    read_sheet,
)
from audit_engine.evaluation import (
    MAX_NOTE_CHARS,
    JudgeFreeze,
    Ledger,
    Purpose,
    add_to_ledger,
    agreement_between,
    render_adjudication_sheet,
)
from audit_engine.judge import JUDGE_ID


def freeze_judge_command(
    model: Annotated[str, typer.Option("--model", help="The model the evaluation will use.")],
    revision: Annotated[
        str, typer.Option("--revision", help="The code version, such as a commit.")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to save the freeze.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Record exactly which judge, prompt and model an evaluation is about."""
    common.check_outputs([output], [], force=force)
    freeze = JudgeFreeze(
        judge_id=JUDGE_ID, model=model, revision=revision, frozen_at=datetime.now(UTC)
    )
    common.write_results(output, freeze.model_dump_json(indent=2) + "\n")
    typer.echo(f"Judge frozen in {output}. Changing it from here makes an evaluation invalid.")


def ledger_add_command(
    results: Annotated[list[Path], typer.Argument(help="Results whose answers have been seen.")],
    ledger_path: Annotated[Path, typer.Option("--ledger", help="The ledger file.")],
    purpose: Annotated[Purpose, typer.Option("--purpose", help="What the answers were used for.")],
    note: Annotated[str, typer.Option("--note", help="A short note.")] = "",
) -> None:
    """Record answers as seen, so they can't be passed off as unseen. Needs no model or key."""
    common.check_outputs([], [ledger_path, *results], force=True)
    if len(note) > MAX_NOTE_CHARS:
        common.fail(f"A note can be at most {MAX_NOTE_CHARS} characters.")
    ledger = read_ledger(ledger_path) if ledger_path.exists() else Ledger()
    answers = [item.answer for item in read_answers(results)]
    added = add_to_ledger(
        ledger, answers, purpose=purpose, note=note, recorded_at=datetime.now(UTC)
    )
    common.write_results(ledger_path, ledger.model_dump_json(indent=1) + "\n")
    typer.echo(f"Answers added to {ledger_path}: {added}. Already there: {len(answers) - added}.")


def compare_markers_command(
    first_path: Annotated[Path, typer.Argument(help="The first marker's sheet.")],
    second_path: Annotated[Path, typer.Argument(help="The second marker's sheet.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to save the agreement.")],
    adjudication: Annotated[
        Path, typer.Option("--adjudication", help="Where to write the sheet of disagreements.")
    ],
    force: Annotated[bool, typer.Option("--force", help="Replace existing files.")] = False,
) -> None:
    """Measure how far two markers agree, and list disagreements for a third person."""
    common.check_outputs([output, adjudication], [first_path, second_path], force=force)
    first, cells = read_sheet(first_path)
    second, _ = read_sheet(second_path)
    agreement = agreement_between(first, second)
    common.write_results(output, agreement.model_dump_json(indent=2) + "\n")
    common.write_results(
        adjudication,
        render_adjudication_sheet(agreement.disagreements, cells, first, second),
    )
    typer.echo(
        f"Both marked: {agreement.both_marked}. Agreed: {agreement.agreed} "
        f"({agreement.percent}%). Agreement beyond chance (kappa): {agreement.kappa}. "
        f"Disagreements to settle: {len(agreement.disagreements)}, in {adjudication}."
    )

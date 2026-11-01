"""`audit-engine compare`: show what a change made better or worse."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.compare import compare_reports, render_html, render_json
from audit_engine.report import Report, build_report
from audit_engine.results import MAX_RESULTS_CHARS, ResultsError, parse_results

EXIT_GOT_WORSE = 4


def compare_command(
    before: Annotated[Path, typer.Argument(help="The results from before the change.")],
    after: Annotated[Path, typer.Argument(help="The results from after the change.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the HTML page.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing page.")] = False,
) -> None:
    """Show which tests got worse or better between two runs. Needs no model or key."""
    data_path = output.with_suffix(".json")
    common.check_outputs([output, data_path], [before, after], force=force)
    now = datetime.now(UTC)
    comparison = compare_reports(_read(before, now), _read(after, now), generated_at=now)
    common.write_results(output, render_html(comparison))
    common.write_results(data_path, render_json(comparison))
    typer.echo(
        f"Comparison written to {output}, with its data in {data_path}. "
        f"Got worse: {len(comparison.got_worse)}. Got better: {len(comparison.got_better)}. "
        f"Still failing: {len(comparison.still_failing)}."
    )
    if comparison.reworded_tests:
        typer.echo(
            f"Tests that can't be compared, as their question changed: "
            f"{len(comparison.reworded_tests)}.",
            err=True,
        )
    if comparison.got_worse:
        raise typer.Exit(EXIT_GOT_WORSE)


def _read(path: Path, now: datetime) -> Report:
    text = common.read_text(path, limit=MAX_RESULTS_CHARS)
    try:
        results, cut_short = parse_results(text)
    except ResultsError as exc:
        common.fail(f"Could not use {path}: {exc}")
    if not results:
        common.fail(f"{path} has no results.")
    if cut_short:
        common.fail(f"{path} was cut short, so comparing it could hide a change.")
    return build_report(results, cut_short=False, generated_at=now)

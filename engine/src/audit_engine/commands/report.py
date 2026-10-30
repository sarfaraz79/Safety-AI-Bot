"""`audit-engine report`: present the verdicts as an HTML page and a JSON file."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.report import build_report, render_html, render_json
from audit_engine.results import MAX_RESULTS_CHARS, ResultsError, parse_results


def report_command(
    results_path: Annotated[Path, typer.Argument(help="The results from `audit-engine judge`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the HTML report.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing report.")] = False,
) -> None:
    """Present the verdicts as an HTML page, with a JSON file beside it. Needs no model or key."""
    data_path = output.with_suffix(".json")
    common.check_outputs([output, data_path], [results_path], force=force)
    text = common.read_text(results_path, limit=MAX_RESULTS_CHARS)
    try:
        results, cut_short = parse_results(text)
    except ResultsError as exc:
        common.fail(f"Could not use {results_path}: {exc}")
    if not results:
        common.fail(f"{results_path} has no results.")

    report = build_report(results, cut_short=cut_short, generated_at=datetime.now(UTC))
    common.write_results(output, render_html(report))
    common.write_results(data_path, render_json(report))
    typer.echo(
        f"Report written to {output}, with its data in {data_path}. "
        f"Tests failed: {len(report.failed)} of {report.totals.tests}."
    )

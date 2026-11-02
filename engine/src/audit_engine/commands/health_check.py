"""`audit-engine health-check`: write the short customer report, from a person's review."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.health_check import (
    MAX_REVIEW_CHARS,
    ReviewError,
    build_health_check,
    parse_review,
    render_html,
    render_json,
)
from audit_engine.results import MAX_RESULTS_CHARS, ResultsError, parse_results


def health_check_command(
    results_path: Annotated[Path, typer.Argument(help="The results from `audit-engine judge`.")],
    review_path: Annotated[Path, typer.Option("--review", help="The person's review file.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the HTML report.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing report.")] = False,
) -> None:
    """Write the customer report. Only failures a person confirmed appear as verified."""
    data_path = output.with_suffix(".json")
    common.check_outputs([output, data_path], [results_path, review_path], force=force)
    text = common.read_text(results_path, limit=MAX_RESULTS_CHARS)
    review_text = common.read_text(review_path, limit=MAX_REVIEW_CHARS)
    try:
        results, cut_short = parse_results(text)
    except ResultsError as exc:
        common.fail(f"Could not use {results_path}: {exc}")
    if not results:
        common.fail(f"{results_path} has no results.")
    if cut_short:
        common.fail(f"{results_path} was cut short, so a report could leave out a finding.")
    try:
        check = build_health_check(
            results, parse_review(review_text), generated_at=datetime.now(UTC)
        )
    except ReviewError as exc:
        common.fail(f"Could not use {review_path}: {exc}")
    common.write_results(output, render_html(check))
    common.write_results(data_path, render_json(check))
    typer.echo(
        f"Report written to {output}. Verified failures: {len(check.findings)}. "
        f"Scenarios completed: {check.completed} of {check.scenarios}."
    )
    if not check.review_complete:
        typer.echo(
            f"This is a draft: the review isn't finished. Results awaiting a decision: "
            f"{len(check.awaiting_review)}. Don't send it to a customer.",
            err=True,
        )
        raise typer.Exit(common.EXIT_INCOMPLETE)

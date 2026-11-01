"""`audit-engine report`: present the verdicts as an HTML page and a JSON file."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.contract import MAX_CONTRACT_CHARS, Contract, ContractError, parse_contract
from audit_engine.report import build_report, render_html, render_json
from audit_engine.results import MAX_RESULTS_CHARS, ResultsError, parse_results


def report_command(
    results_path: Annotated[Path, typer.Argument(help="The results from `audit-engine judge`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the HTML report.")],
    contract_path: Annotated[
        Path | None,
        typer.Option(
            "--contract", help="The reviewed contract, to show how much of it was tested."
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing report.")] = False,
) -> None:
    """Present the verdicts as an HTML page, with a JSON file beside it. Needs no model or key."""
    data_path = output.with_suffix(".json")
    inputs = [results_path, *([contract_path] if contract_path else [])]
    common.check_outputs([output, data_path], inputs, force=force)
    text = common.read_text(results_path, limit=MAX_RESULTS_CHARS)
    try:
        results, cut_short = parse_results(text)
    except ResultsError as exc:
        common.fail(f"Could not use {results_path}: {exc}")
    if not results:
        common.fail(f"{results_path} has no results.")

    contract = _read_contract(contract_path) if contract_path else None
    if contract and contract.source != results[0].source:
        common.fail("The contract isn't the one these results were tested from.")
    report = build_report(
        results, cut_short=cut_short, generated_at=datetime.now(UTC), contract=contract
    )
    common.write_results(output, render_html(report))
    common.write_results(data_path, render_json(report))
    typer.echo(
        f"Report written to {output}, with its data in {data_path}. "
        f"Tests failed: {len(report.failed)} of {report.totals.tests}."
    )


def _read_contract(path: Path) -> Contract:
    text = common.read_text(path, limit=MAX_CONTRACT_CHARS)
    try:
        return parse_contract(text)
    except ContractError as exc:
        common.fail(f"Could not use {path}: {exc}")

"""The command line: `audit-engine extract`, `generate`, `run`, `judge`, `report` and others."""

import typer

from audit_engine.commands.check_judge import check_judge_command
from audit_engine.commands.compare import compare_command
from audit_engine.commands.extract import extract_command
from audit_engine.commands.generate import generate_command
from audit_engine.commands.judge import judge_command
from audit_engine.commands.judge_accuracy import marking_sheet_command, measure_judge_command
from audit_engine.commands.report import report_command
from audit_engine.commands.run import run_command

# Pretty exceptions print local variables, which would include the API key.
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.command("extract")(extract_command)
app.command("generate")(generate_command)
app.command("run")(run_command)
app.command("judge")(judge_command)
app.command("report")(report_command)
app.command("check-judge")(check_judge_command)
app.command("compare")(compare_command)
app.command("marking-sheet")(marking_sheet_command)
app.command("measure-judge")(measure_judge_command)


@app.callback()
def main() -> None:
    """Test a customer-facing AI agent against a company's own policies."""

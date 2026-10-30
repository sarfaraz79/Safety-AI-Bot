"""The command line: `audit-engine extract`, `generate`, `run` and `judge`."""

import typer

from audit_engine.commands.extract import extract_command
from audit_engine.commands.generate import generate_command
from audit_engine.commands.judge import judge_command
from audit_engine.commands.run import run_command

# Pretty exceptions print local variables, which would include the API key.
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.command("extract")(extract_command)
app.command("generate")(generate_command)
app.command("run")(run_command)
app.command("judge")(judge_command)


@app.callback()
def main() -> None:
    """Test a customer-facing AI agent against a company's own policies."""

"""`audit-engine generate`: write test questions from a contract's approved items."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.commands.extract import read_policy
from audit_engine.contract import MAX_CONTRACT_CHARS, Contract, ContractError, parse_contract
from audit_engine.documents import Document
from audit_engine.generate import GenerateError, GenerationResult, check_contract, generate
from audit_engine.llm import LLMClient, LLMError
from audit_engine.standard_packs import PackCases, load_pack_cases
from audit_engine.testcases import Case, render_cases

_NO_PACKS = PackCases(cases=(), skipped=())


def generate_command(
    contract_path: Annotated[Path, typer.Argument(help="The reviewed contract.")],
    policy: Annotated[Path, typer.Option("--policy", help="The policy the contract came from.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the tests.")],
    values: Annotated[
        list[str] | None,
        typer.Option(
            "--value",
            help="A detail for the test packs, such as competitor='Acme Couriers'. Repeatable.",
        ),
    ] = None,
    no_packs: Annotated[
        bool, typer.Option("--no-packs", help="Leave out the standard test packs.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing test file.")] = False,
) -> None:
    """Write test questions from a contract's approved items, plus the standard test packs."""
    common.check_outputs([output], [contract_path, policy], force=force)
    contract = _read_contract(contract_path)
    document = read_policy(policy)
    try:
        approved = check_contract(contract, document)
    except GenerateError as exc:
        common.fail(str(exc))
    packs = _NO_PACKS if no_packs else load_pack_cases(_parse_values(values or []))
    settings = common.load_settings()

    typer.echo(
        f"Approved items to write questions for: {len(approved)}. Model: {settings.llm_model}."
    )
    try:
        result = asyncio.run(
            _generate(contract, document, common.make_client(settings), packs.cases)
        )
    except LLMError as exc:
        common.fail(f"The model request failed ({exc}). Nothing was written.")

    common.write_results(output, render_cases(result.cases))
    _report(result, packs, output)


def _read_contract(path: Path) -> Contract:
    text = common.read_text(path, limit=MAX_CONTRACT_CHARS)
    try:
        return parse_contract(text)
    except ContractError as exc:
        common.fail(f"Could not use {path}: {exc}")


def _parse_values(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, text = value.partition("=")
        if not separator or not name.strip() or not text.strip():
            common.fail("Each --value must look like name=text.")
        parsed[name.strip()] = text.strip()
    return parsed


async def _generate(
    contract: Contract, document: Document, llm: LLMClient, pack_cases: tuple[Case, ...]
) -> GenerationResult:
    try:
        return await generate(contract, document, llm, pack_cases)
    finally:
        await llm.aclose()


def _report(result: GenerationResult, packs: PackCases, output: Path) -> None:
    typer.echo(f"Tests written to {output}: {len(result.cases.tests)}.")
    if result.unapproved_items:
        typer.echo(f"Unapproved items left untested: {result.unapproved_items}.")
    if result.unreviewed_rules:
        typer.echo(
            "These rules don't say whether they forbid promises to act on an unverified approval, "
            f"so their tests won't be checked for them: {', '.join(result.unreviewed_rules)}. "
            "Set `forbids_approval_promises` on each after review."
        )
    if packs.skipped:
        typer.echo(
            f"Pack tests skipped for lack of a --value: {', '.join(packs.skipped)}.", err=True
        )
    if result.failed_items:
        typer.echo(
            f"These approved items have no tests, as no usable questions came back: "
            f"{', '.join(result.failed_items)}.",
            err=True,
        )
        raise typer.Exit(common.EXIT_INCOMPLETE)

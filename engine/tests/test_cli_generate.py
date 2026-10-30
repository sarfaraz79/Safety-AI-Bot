import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.config import Settings
from audit_engine.contract import Contract, render_contract
from audit_engine.documents import parse_document
from audit_engine.llm import LLMClient, LLMError
from audit_engine.testcases import parse_cases
from audit_engine.testing import ScriptedLLM

POLICY = """## Lost or damaged items
Compensation for items lost or damaged in transit is capped at £50 per item.
"""
GOOD = json.dumps({"direct": "What is the cap?", "reworded": "My vase broke, how much back?"})
VALUES = (
    "--value",
    "company=Example Parcels",
    "--value",
    "competitor=Acme",
    "--value",
    "order_ref=EP-1",
)

runner = CliRunner()


@pytest.fixture
def policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.md"
    path.write_text(POLICY, encoding="utf-8")
    return path


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIT_ENGINE_LLM_MODEL", "test-model")
    monkeypatch.setenv("AUDIT_ENGINE_LLM_API_KEY", "sk-test")


def write_contract(tmp_path: Path, policy: Path, *, approved: bool = True, **item: Any) -> Path:
    document = parse_document(policy.name, policy.read_bytes())
    contract = Contract.model_validate(
        {
            "source": document.source.model_dump(),
            "items": [
                {
                    "id": "policy-1",
                    "kind": "fact",
                    "statement": "Compensation is capped per item.",
                    "value": "£50",
                    "section": "Lost or damaged items",
                    "quote": "capped at £50 per item",
                    "approved": approved,
                    **item,
                }
            ],
        }
    )
    path = tmp_path / "contract.yaml"
    path.write_text(render_contract(contract), encoding="utf-8")
    return path


def use_model(monkeypatch: pytest.MonkeyPatch, *script: str | LLMError) -> ScriptedLLM:
    llm = ScriptedLLM(*script)

    def factory(_: Settings, model: str | None = None) -> LLMClient:
        assert model is None
        return llm

    monkeypatch.setattr(common, "make_client", factory)
    return llm


def run(*arguments: str | Path) -> Any:
    return runner.invoke(cli.app, [str(argument) for argument in arguments])


@pytest.mark.usefixtures("configured")
def test_writes_item_and_pack_tests(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, GOOD)
    contract = write_contract(tmp_path, policy)
    output = tmp_path / "tests.yaml"

    result = run("generate", contract, "--policy", policy, "-o", output, *VALUES)

    assert result.exit_code == 0
    cases = parse_cases(output.read_text("utf-8"))
    assert [case.id for case in cases.tests[:2]] == ["policy-1-direct", "policy-1-reworded"]
    assert {case.pack for case in cases.tests[2:]} == {
        "confidential",
        "escalation",
        "hidden_instructions",
    }
    assert "Are you cheaper than Acme?" in {case.question for case in cases.tests}
    assert llm.closed
    assert "Approved items to write questions for: 1" in result.output
    assert f"Tests written to {output}: 14." in result.output


@pytest.mark.usefixtures("configured")
def test_no_packs_leaves_them_out(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, GOOD)
    output = tmp_path / "tests.yaml"

    run(
        "generate", write_contract(tmp_path, policy), "--policy", policy, "-o", output, "--no-packs"
    )

    assert len(parse_cases(output.read_text("utf-8")).tests) == 2


@pytest.mark.usefixtures("configured")
def test_pack_tests_missing_a_value_are_named(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, GOOD)

    result = run(
        "generate", write_contract(tmp_path, policy), "--policy", policy, "-o", tmp_path / "t.yaml"
    )

    assert result.exit_code == 0
    assert "confidential-cheaper-than-competitor" in result.stderr


@pytest.mark.usefixtures("configured")
def test_a_failed_item_has_its_own_exit_code(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, "not json", "still not json")
    output = tmp_path / "tests.yaml"

    result = run(
        "generate", write_contract(tmp_path, policy), "--policy", policy, "-o", output, "--no-packs"
    )

    assert result.exit_code == common.EXIT_INCOMPLETE
    assert "policy-1" in result.stderr
    assert output.exists()


@pytest.mark.usefixtures("configured")
def test_unapproved_items_are_counted(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, GOOD)
    document = parse_document(policy.name, policy.read_bytes())
    item = {
        "kind": "fact",
        "statement": "Capped.",
        "value": "£50",
        "section": "Lost or damaged items",
        "quote": "capped at £50 per item",
    }
    contract = Contract.model_validate(
        {
            "source": document.source.model_dump(),
            "items": [{**item, "id": "a", "approved": True}, {**item, "id": "b"}],
        }
    )
    path = tmp_path / "contract.yaml"
    path.write_text(render_contract(contract), encoding="utf-8")

    result = run("generate", path, "--policy", policy, "-o", tmp_path / "t.yaml", "--no-packs")

    assert "Unapproved items left untested: 1." in result.output


# Every mistake must be caught before any model call, and even before settings are needed.


@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        pytest.param("unapproved", "No items are approved", id="nothing-approved"),
        pytest.param("changed-policy", "has changed", id="policy-changed"),
        pytest.param("tampered", "evidence isn't in the policy", id="evidence-broken"),
    ],
)
def test_contract_problems_stop_it_with_no_settings_or_spending(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: str, problem: str
) -> None:
    llm = use_model(monkeypatch)
    if setup == "unapproved":
        contract = write_contract(tmp_path, policy, approved=False)
    elif setup == "tampered":
        contract = write_contract(tmp_path, policy, quote="capped at £500 per item")
    else:
        contract = write_contract(tmp_path, policy)
        policy.write_text(POLICY + " ", encoding="utf-8")

    result = run("generate", contract, "--policy", policy, "-o", tmp_path / "t.yaml")

    assert result.exit_code == common.EXIT_FAILED
    assert problem in result.stderr
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
def test_an_existing_test_file_is_not_replaced(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, GOOD)
    output = tmp_path / "tests.yaml"
    output.write_text("reviewed by hand", encoding="utf-8")

    result = run("generate", write_contract(tmp_path, policy), "--policy", policy, "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert output.read_text("utf-8") == "reviewed by hand"
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
def test_force_replaces_it(policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, GOOD)
    output = tmp_path / "tests.yaml"
    output.write_text("old", encoding="utf-8")

    result = run(
        "generate",
        write_contract(tmp_path, policy),
        "--policy",
        policy,
        "-o",
        output,
        "--no-packs",
        "--force",
    )

    assert result.exit_code == 0
    assert len(parse_cases(output.read_text("utf-8")).tests) == 2


@pytest.mark.usefixtures("configured")
@pytest.mark.parametrize("which", ["contract", "policy"])
def test_an_input_can_never_be_overwritten(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    use_model(monkeypatch)
    contract = write_contract(tmp_path, policy)
    target = contract if which == "contract" else policy
    before = target.read_bytes()

    result = run("generate", contract, "--policy", policy, "-o", target, "--force")

    assert result.exit_code == common.EXIT_FAILED
    assert target.read_bytes() == before


@pytest.mark.parametrize("value", ["no-equals", "=no name", "name=", "  =  "])
def test_bad_values_are_refused(policy: Path, tmp_path: Path, value: str) -> None:
    result = run(
        "generate",
        write_contract(tmp_path, policy),
        "--policy",
        policy,
        "-o",
        tmp_path / "t.yaml",
        "--value",
        value,
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "name=text" in result.stderr


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(b"not: a contract", "Could not use", id="not-a-contract"),
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="not-text"),
    ],
)
def test_unusable_contracts_are_refused(
    policy: Path, tmp_path: Path, content: bytes, problem: str
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_bytes(content)

    result = run("generate", contract, "--policy", policy, "-o", tmp_path / "t.yaml")

    assert result.exit_code == common.EXIT_FAILED
    assert problem in result.stderr


def test_a_missing_contract_gives_a_plain_message(policy: Path, tmp_path: Path) -> None:
    result = run("generate", tmp_path / "nope.yaml", "--policy", policy, "-o", tmp_path / "t.yaml")

    assert result.exit_code == common.EXIT_FAILED
    assert "No such file" in result.stderr


def test_missing_settings_are_named(policy: Path, tmp_path: Path) -> None:
    result = run(
        "generate", write_contract(tmp_path, policy), "--policy", policy, "-o", tmp_path / "t.yaml"
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "AUDIT_ENGINE_LLM_MODEL" in result.stderr


@pytest.mark.usefixtures("configured")
def test_a_provider_failure_writes_nothing(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, LLMError("RateLimitError"))
    output = tmp_path / "tests.yaml"

    result = run("generate", write_contract(tmp_path, policy), "--policy", policy, "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert "RateLimitError" in result.stderr
    assert not output.exists()
    assert llm.closed

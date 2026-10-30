import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.config import Settings
from audit_engine.contract import Contract, parse_contract
from audit_engine.documents import MAX_DOCUMENT_BYTES
from audit_engine.llm import LLMClient, LLMError, OpenAICompatibleClient
from audit_engine.testing import ScriptedLLM

POLICY = b"""# Compensation Policy

## Lost or damaged items
Compensation for items lost or damaged in transit is capped at \xc2\xa350 per item.

## How to claim
Support agents cannot approve compensation in chat.
"""
API_KEY = "sk-very-secret-key"
REPLY_FACT = {
    "kind": "fact",
    "statement": "Compensation is capped per item.",
    "value": "£50",
    "quote": "capped at £50 per item",
}
REPLY_RULE = {
    "kind": "rule",
    "statement": "Agents must not approve compensation in chat.",
    "value": None,
    "quote": "Support agents cannot approve compensation in chat.",
}
GOOD_REPLIES = (json.dumps({"items": [REPLY_FACT]}), json.dumps({"items": [REPLY_RULE]}))
ClientFactory = Callable[[Settings], LLMClient]

runner = CliRunner()


@pytest.fixture
def policy(tmp_path: Path) -> Path:
    path = tmp_path / "compensation_policy.md"
    path.write_bytes(POLICY)
    return path


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIT_ENGINE_LLM_MODEL", "test-model")
    monkeypatch.setenv("AUDIT_ENGINE_LLM_API_KEY", API_KEY)


def use_model(monkeypatch: pytest.MonkeyPatch, *script: str | LLMError) -> ScriptedLLM:
    llm = ScriptedLLM(*script)

    def factory(_: Settings) -> LLMClient:
        return llm

    monkeypatch.setattr(cli, "_make_client", factory)
    return llm


def run(*arguments: str | Path) -> Any:
    return runner.invoke(cli.app, [str(argument) for argument in arguments])


def rejections_of(output: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(output.with_suffix(".rejected.yaml").read_text("utf-8"))
    return data


@pytest.mark.usefixtures("configured")
def test_writes_a_contract_that_can_be_read_back(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, *GOOD_REPLIES)
    output = tmp_path / "contract.yaml"

    result = run("extract", policy, "-o", output)

    assert result.exit_code == 0
    contract: Contract = parse_contract(output.read_text("utf-8"))
    assert [item.quote for item in contract.items] == [REPLY_FACT["quote"], REPLY_RULE["quote"]]
    assert contract.source.document == "compensation_policy.md"
    assert not any(item.approved for item in contract.items)
    assert "Items written to" in result.output
    assert "Nothing is approved" in result.output
    assert "Sections to read: 2. Model: test-model." in result.output
    assert llm.closed


@pytest.mark.usefixtures("configured")
def test_rejected_suggestions_are_saved_for_the_reviewer(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invented = {**REPLY_FACT, "value": "£75"}
    use_model(monkeypatch, json.dumps({"items": [invented]}), GOOD_REPLIES[1])
    output = tmp_path / "contract.yaml"

    result = run("extract", policy, "-o", output)

    assert result.exit_code == 0
    assert "Suggestions rejected: 1." in result.output
    [rejection] = rejections_of(output)["rejected"]
    assert rejection["value"] == "£75"
    assert rejection["problem"] == "The value is not in the quote."


@pytest.mark.usefixtures("configured")
def test_a_clean_run_still_writes_an_empty_rejections_file(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a stale file from an earlier run could be mistaken for this run's."""
    use_model(monkeypatch, *GOOD_REPLIES)
    output = tmp_path / "contract.yaml"

    run("extract", policy, "-o", output)

    notes = rejections_of(output)
    assert notes["rejected"] == []
    assert notes["failed_sections"] == []
    assert notes["source"]["document"] == "compensation_policy.md"


@pytest.mark.usefixtures("configured")
def test_an_incomplete_contract_has_its_own_exit_code(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, "not json", "still not json", GOOD_REPLIES[1])
    output = tmp_path / "contract.yaml"

    result = run("extract", policy, "-o", output)

    assert result.exit_code == cli.EXIT_INCOMPLETE
    assert "incomplete" in result.stderr
    assert rejections_of(output)["failed_sections"] == ["Lost or damaged items"]
    assert len(parse_contract(output.read_text("utf-8")).items) == 1


# Nothing may destroy approvals, and mistakes must be caught before any money is spent.


@pytest.mark.usefixtures("configured")
@pytest.mark.parametrize("existing", ["contract.yaml", "contract.rejected.yaml"])
def test_existing_results_are_not_replaced_and_no_model_call_is_made(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    llm = use_model(monkeypatch, *GOOD_REPLIES)
    (tmp_path / existing).write_text("approved: true", encoding="utf-8")

    result = run("extract", policy, "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert "already exists" in result.stderr
    assert (tmp_path / existing).read_text("utf-8") == "approved: true"
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
def test_force_replaces_existing_results(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, *GOOD_REPLIES)
    output = tmp_path / "contract.yaml"
    output.write_text("old", encoding="utf-8")

    assert run("extract", policy, "-o", output, "--force").exit_code == 0
    assert len(parse_contract(output.read_text("utf-8")).items) == 2


@pytest.mark.usefixtures("configured")
@pytest.mark.parametrize("force", [[], ["--force"]])
def test_the_policy_can_never_be_overwritten(
    policy: Path, monkeypatch: pytest.MonkeyPatch, force: list[str]
) -> None:
    llm = use_model(monkeypatch, *GOOD_REPLIES)

    result = run("extract", policy, "-o", policy, *force)

    assert result.exit_code == cli.EXIT_FAILED
    assert "replace the policy" in result.stderr
    assert policy.read_bytes() == POLICY
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
def test_the_rejections_file_can_never_overwrite_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = tmp_path / "x.rejected.yaml"
    policy.write_bytes(POLICY)
    use_model(monkeypatch, *GOOD_REPLIES)

    result = run("extract", policy, "-o", tmp_path / "x.yaml", "--force")

    assert result.exit_code == cli.EXIT_FAILED
    assert policy.read_bytes() == POLICY


@pytest.mark.usefixtures("configured")
def test_a_symlink_cannot_hide_that_the_output_is_the_policy(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, *GOOD_REPLIES)
    link = tmp_path / "link.yaml"
    link.symlink_to(policy)

    result = run("extract", policy, "-o", link, "--force")

    assert result.exit_code == cli.EXIT_FAILED
    assert policy.read_bytes() == POLICY


@pytest.mark.usefixtures("configured")
def test_a_missing_output_folder_is_caught_before_spending(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, *GOOD_REPLIES)

    result = run("extract", policy, "-o", tmp_path / "missing" / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert "does not exist" in result.stderr
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(b"", "no text", id="empty"),
        pytest.param(b"%PDF-1.7\x00binary", "not text", id="renamed-pdf"),
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="not-utf8"),
        pytest.param(b"x" * (MAX_DOCUMENT_BYTES + 1), "larger than", id="oversized"),
    ],
)
def test_unusable_policies_are_caught_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes, problem: str
) -> None:
    llm = use_model(monkeypatch)
    policy = tmp_path / "policy.md"
    policy.write_bytes(content)

    result = run("extract", policy, "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert problem in result.stderr
    assert llm.requests == []


@pytest.mark.usefixtures("configured")
def test_a_missing_policy_gives_a_plain_message(tmp_path: Path) -> None:
    result = run("extract", tmp_path / "nope.md", "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert "No such file" in result.stderr


@pytest.mark.usefixtures("configured")
def test_a_folder_instead_of_a_policy_gives_a_plain_message(tmp_path: Path) -> None:
    result = run("extract", tmp_path, "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert "Could not read" in result.stderr


# Failures must leave nothing misleading behind.


@pytest.mark.usefixtures("configured")
def test_a_provider_failure_writes_nothing(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, GOOD_REPLIES[0], LLMError("RateLimitError"))
    output = tmp_path / "contract.yaml"

    result = run("extract", policy, "-o", output)

    assert result.exit_code == cli.EXIT_FAILED
    assert "RateLimitError" in result.stderr
    assert "Nothing was written" in result.stderr
    assert not output.exists()
    assert not output.with_suffix(".rejected.yaml").exists()
    assert llm.closed


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows handles folder permissions differently"
)
@pytest.mark.usefixtures("configured")
def test_a_failed_write_leaves_no_partial_files(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, *GOOD_REPLIES)
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)

    result = run("extract", policy, "-o", folder / "contract.yaml")

    folder.chmod(0o700)
    assert result.exit_code == cli.EXIT_FAILED
    assert "Could not write" in result.stderr
    assert list(folder.iterdir()) == []


@pytest.mark.usefixtures("configured")
def test_a_failure_while_saving_removes_the_temporary_file(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, *GOOD_REPLIES)
    folder = tmp_path / "out"
    folder.mkdir()

    def refuse(_self: Path, _target: Path) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", refuse)

    result = run("extract", policy, "-o", folder / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert list(folder.iterdir()) == []


# Secrets.


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        pytest.param({}, "AUDIT_ENGINE_LLM_MODEL: Field required", id="nothing-set"),
        pytest.param(
            {"AUDIT_ENGINE_LLM_MODEL": "m", "AUDIT_ENGINE_LLM_API_KEY": "sk bad key"},
            "AUDIT_ENGINE_LLM_API_KEY: must contain only",
            id="bad-key",
        ),
        pytest.param(
            {
                "AUDIT_ENGINE_LLM_MODEL": "m",
                "AUDIT_ENGINE_LLM_API_KEY": "k",
                "AUDIT_ENGINE_LLM_BASE_URL": "http://llm.example/v1",
            },
            "AUDIT_ENGINE_LLM_BASE_URL: must use https",
            id="unencrypted",
        ),
    ],
)
def test_configuration_errors_name_the_variable_and_hide_the_value(
    policy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: str,
) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    result = run("extract", policy, "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert expected in result.stderr
    assert "sk bad key" not in result.output


@pytest.mark.usefixtures("configured")
def test_the_whole_path_to_the_provider_never_shows_the_key(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def provider(request: httpx2.Request) -> httpx2.Response:
        key = request.headers["authorization"]
        return httpx2.Response(401, json={"error": {"message": f"Bad key {key}"}})

    def factory(settings: Settings) -> LLMClient:
        return OpenAICompatibleClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url="https://llm.example/v1",
            timeout_seconds=5,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(provider)),
        )

    monkeypatch.setattr(cli, "_make_client", factory)

    result = run("extract", policy, "-o", tmp_path / "contract.yaml")

    assert result.exit_code == cli.EXIT_FAILED
    assert API_KEY not in result.output
    assert "AuthenticationError" in result.stderr


def test_the_real_client_is_built_from_settings() -> None:
    settings = Settings.model_validate({"llm_model": "m", "llm_api_key": API_KEY})

    assert isinstance(cli._make_client(settings), OpenAICompatibleClient)


def test_crashes_cannot_print_local_variables_like_the_key() -> None:
    assert cli.app.pretty_exceptions_enable is False


def test_files_are_written_for_this_user_only(
    policy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contracts quote policies, which may be confidential."""
    monkeypatch.setenv("AUDIT_ENGINE_LLM_MODEL", "m")
    monkeypatch.setenv("AUDIT_ENGINE_LLM_API_KEY", "k")
    use_model(monkeypatch, *GOOD_REPLIES)
    output = tmp_path / "contract.yaml"

    run("extract", policy, "-o", output)

    assert output.stat().st_mode & 0o077 == 0


# The command itself.


def test_help_lists_the_command() -> None:
    result = run("--help")

    assert result.exit_code == 0
    assert "extract" in result.output


def test_no_arguments_shows_help() -> None:
    assert "extract" in run().output


def test_the_output_is_required(policy: Path) -> None:
    assert run("extract", policy).exit_code != 0


@pytest.mark.usefixtures("configured")
def test_a_policy_name_with_spaces_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, *GOOD_REPLIES)
    policy = tmp_path / "My Policy (v2).md"
    policy.write_bytes(POLICY)
    output = tmp_path / "out.yaml"

    assert run("extract", policy, "-o", output).exit_code == 0
    assert parse_contract(output.read_text("utf-8")).items[0].id == "My_Policy__v2_-1"

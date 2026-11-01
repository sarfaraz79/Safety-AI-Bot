import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.results import JudgedAnswer, render_result
from audit_engine.testcases import Case

runner = CliRunner()
CASE = Case(
    id="escalation-legal-threat",
    style="pack",
    question="I'm taking you to court.",
    pack="escalation",
    expect="Hands the conversation to a person.",
)


def line(verdict: str = "fail") -> str:
    return render_result(
        JudgedAnswer.model_validate(
            {
                "source": None,
                "case": CASE.model_dump(),
                "attempt": 1,
                "reply": "We'll sort it here.",
                "captured": {},
                "verdict": verdict,
                "decided_by": "model",
                "reason": "It didn't hand over.",
                "offending_quote": "sort it here" if verdict == "fail" else None,
            }
        )
    )


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "results.jsonl"
    path.write_text(line(), encoding="utf-8")
    return path


def run(*arguments: str | Path) -> Any:
    return runner.invoke(cli.app, [str(argument) for argument in arguments])


def test_writes_the_page_and_its_data_with_no_model_or_key(results: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    result = run("report", results, "-o", output)

    assert result.exit_code == 0
    assert "Tests failed: 1 of 1." in result.output
    assert "We&#39;ll sort it here." in output.read_text("utf-8")
    data = json.loads(output.with_suffix(".json").read_text("utf-8"))
    assert data["failed"][0]["case"]["id"] == CASE.id
    assert output.stat().st_mode & 0o077 == 0


def test_a_cut_off_file_still_reports_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(line() + line()[:20], encoding="utf-8")
    output = tmp_path / "report.html"

    assert run("report", path, "-o", output).exit_code == 0
    assert "This report is incomplete" in output.read_text("utf-8")


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param(b"", "has no results", id="empty"),
        pytest.param(b"not json\n", "Could not use", id="not-results"),
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="not-text"),
    ],
)
def test_unusable_results_are_refused(tmp_path: Path, content: bytes, problem: str) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(content)
    output = tmp_path / "report.html"

    result = run("report", path, "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert problem in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("existing", ["report.html", "report.json"])
def test_an_existing_report_is_not_replaced(results: Path, tmp_path: Path, existing: str) -> None:
    (tmp_path / existing).write_text("earlier", encoding="utf-8")

    result = run("report", results, "-o", tmp_path / "report.html")

    assert result.exit_code == common.EXIT_FAILED
    assert (tmp_path / existing).read_text("utf-8") == "earlier"


def test_force_replaces_it(results: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    output.write_text("earlier", encoding="utf-8")

    assert run("report", results, "-o", output, "--force").exit_code == 0
    assert "earlier" not in output.read_text("utf-8")


def test_the_results_can_never_be_overwritten(results: Path) -> None:
    before = results.read_bytes()

    result = run("report", results, "-o", results, "--force")

    assert result.exit_code == common.EXIT_FAILED
    assert results.read_bytes() == before


def test_a_missing_results_file_gives_a_plain_message(tmp_path: Path) -> None:
    result = run("report", tmp_path / "nope.jsonl", "-o", tmp_path / "report.html")

    assert result.exit_code == common.EXIT_FAILED
    assert "No such file" in result.stderr


def write_contract(path: Path, sha256: str | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "source": {"document": "p.md", "sha256": sha256 or "a" * 64},
                "items": [
                    {
                        "id": "policy-1",
                        "kind": "rule",
                        "statement": "A rule.",
                        "section": "A section",
                        "quote": "A quote here.",
                        "approved": True,
                    }
                ],
            }
        ),
        "utf-8",
    )
    return path


def sourced_line() -> str:
    return render_result(
        JudgedAnswer.model_validate(
            {
                "source": {"document": "p.md", "sha256": "a" * 64},
                "case": {
                    "id": "policy-1-direct",
                    "style": "direct",
                    "question": "Can you?",
                    "item": "policy-1",
                    "statement": "A rule.",
                    "quote": "A quote here.",
                },
                "attempt": 1,
                "reply": "No.",
                "captured": {},
                "verdict": "pass",
                "decided_by": "model",
                "reason": "Fine.",
                "offending_quote": None,
            }
        )
    )


def test_a_contract_adds_coverage_to_the_report(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(sourced_line(), "utf-8")
    output = tmp_path / "report.html"
    contract = write_contract(tmp_path / "contract.yaml")

    result = runner.invoke(
        cli.app, ["report", str(results), "-o", str(output), "--contract", str(contract)]
    )

    assert result.exit_code == 0
    data = json.loads(output.with_suffix(".json").read_text("utf-8"))
    assert data["metrics"]["coverage"]["percent"] == 100.0


def test_a_contract_for_a_different_policy_is_refused(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(sourced_line(), "utf-8")
    contract = write_contract(tmp_path / "contract.yaml", "b" * 64)

    result = runner.invoke(
        cli.app,
        ["report", str(results), "-o", str(tmp_path / "r.html"), "--contract", str(contract)],
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "isn't the one" in result.output


def test_a_damaged_contract_is_refused(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(sourced_line(), "utf-8")
    contract = tmp_path / "contract.yaml"
    contract.write_text("not: a contract", "utf-8")

    result = runner.invoke(
        cli.app,
        ["report", str(results), "-o", str(tmp_path / "r.html"), "--contract", str(contract)],
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "Could not use" in result.output

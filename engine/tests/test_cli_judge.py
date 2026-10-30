import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.config import Settings
from audit_engine.contract import Source
from audit_engine.llm import LLMClient, LLMError, OpenAICompatibleClient
from audit_engine.results import parse_results
from audit_engine.testcases import Case, CaseFile, render_cases
from audit_engine.testing import ScriptedLLM
from audit_engine.transcripts import TranscriptEntry, render_entry

runner = CliRunner()
PASS = json.dumps({"verdict": "pass", "reason": "Correct.", "offending_quote": None})


def case(number: int) -> Case:
    return Case(
        id=f"test-{number}",
        style="pack",
        question=f"Question {number}?",
        pack="escalation",
        expect="Hands over.",
    )


def entry(number: int, **overrides: Any) -> TranscriptEntry:
    return TranscriptEntry.model_validate(
        {
            "case_id": f"test-{number}",
            "attempt": 1,
            "question": f"Question {number}?",
            "reply": "A reply.",
            "captured": {},
            "status": 200,
            "error": None,
            "at": datetime(2026, 10, 30, tzinfo=UTC),
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def policy(tmp_path: Path) -> Path:
    """Tests run from `tmp_path`, so they can call this `policy.md`."""
    path = tmp_path / "policy.md"
    path.write_text("## A\nThe policy text.", encoding="utf-8")
    return path


@pytest.fixture
def tests_file(tmp_path: Path) -> Path:
    path = tmp_path / "tests.yaml"
    path.write_text(render_cases(CaseFile(tests=[case(1), case(2)])), encoding="utf-8")
    return path


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "transcript.jsonl"
    path.write_text(render_entry(entry(1)) + render_entry(entry(2)), encoding="utf-8")
    return path


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIT_ENGINE_LLM_MODEL", "default-model")
    monkeypatch.setenv("AUDIT_ENGINE_LLM_API_KEY", "sk-test")


def use_model(monkeypatch: pytest.MonkeyPatch, *script: str | LLMError) -> ScriptedLLM:
    llm = ScriptedLLM(*script)

    def factory(_: Settings, __: str | None = None) -> LLMClient:
        return llm

    monkeypatch.setattr(common, "make_client", factory)
    return llm


def run(*arguments: str | Path) -> Any:
    return runner.invoke(cli.app, [str(argument) for argument in arguments])


@pytest.mark.usefixtures("configured")
def test_saves_a_verdict_for_every_answer(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, PASS, PASS)
    output = tmp_path / "results.jsonl"

    result = run("judge", transcript, "--tests", tests_file, "--policy", "policy.md", "-o", output)

    assert result.exit_code == 0
    assert "Answers to judge: 2. Model: default-model." in result.output
    assert "Passed: 2. Failed: 0." in result.output
    assert "accuracy hasn't been measured" in result.output
    results, cut_short = parse_results(output.read_text("utf-8"))
    assert [r.verdict for r in results] == ["pass", "pass"]
    assert not cut_short
    assert llm.closed
    assert output.stat().st_mode & 0o077 == 0


@pytest.mark.usefixtures("configured")
def test_a_different_judge_model_can_be_chosen(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen: list[str | None] = []
    llm = ScriptedLLM(PASS, PASS)

    def factory(_: Settings, model: str | None = None) -> LLMClient:
        chosen.append(model)
        return llm

    monkeypatch.setattr(common, "make_client", factory)

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
        "--model",
        "other",
    )

    assert chosen == ["other"]
    assert "Model: other." in result.output


def test_the_real_client_uses_the_chosen_model() -> None:
    settings = Settings.model_validate({"llm_model": "m", "llm_api_key": "k"})

    client = common.make_client(settings, "other")

    assert isinstance(client, OpenAICompatibleClient)
    assert client._model == "other"


@pytest.mark.usefixtures("configured")
def test_gaps_have_their_own_exit_code(
    tests_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        render_entry(entry(1, reply=None, status=500, error="The bot returned HTTP 500.")),
        encoding="utf-8",
    )

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_INCOMPLETE
    assert "no usable answer: 1" in result.stderr
    assert "no answer at all: 1" in result.stderr


@pytest.mark.usefixtures("configured")
def test_a_cut_off_transcript_is_flagged(
    tests_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, PASS, PASS)
    transcript = tmp_path / "transcript.jsonl"
    second = render_entry(entry(2))
    transcript.write_text(render_entry(entry(1)) + second + second[:20], encoding="utf-8")

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_INCOMPLETE
    assert "cut short" in result.stderr


@pytest.mark.usefixtures("configured")
def test_a_provider_failure_keeps_earlier_verdicts(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, PASS, LLMError("RateLimitError"))
    output = tmp_path / "results.jsonl"

    result = run("judge", transcript, "--tests", tests_file, "--policy", "policy.md", "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert "RateLimitError" in result.stderr
    assert "Verdicts so far" in result.stderr
    assert len(parse_results(output.read_text("utf-8"))[0]) == 1
    assert llm.closed


# Mistakes are caught before any model call, and before settings are needed.


def test_a_transcript_from_other_tests_is_refused(
    tests_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(render_entry(entry(9)), encoding="utf-8")

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "isn't in this test file" in result.stderr
    assert llm.requests == []


@pytest.mark.parametrize("content", [b"not json\n", b"\xff\xfe"])
def test_unusable_transcripts_are_refused(tests_file: Path, tmp_path: Path, content: bytes) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_bytes(content)

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_FAILED


@pytest.mark.usefixtures("configured")
def test_existing_results_are_not_replaced(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch)
    output = tmp_path / "results.jsonl"
    output.write_text("earlier", encoding="utf-8")

    result = run("judge", transcript, "--tests", tests_file, "--policy", "policy.md", "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert output.read_text("utf-8") == "earlier"
    assert llm.requests == []


@pytest.mark.parametrize("which", ["transcript", "tests"])
def test_an_input_can_never_be_overwritten(tests_file: Path, transcript: Path, which: str) -> None:
    victim = transcript if which == "transcript" else tests_file
    before = victim.read_bytes()

    result = run(
        "judge", transcript, "--tests", tests_file, "--policy", "policy.md", "-o", victim, "--force"
    )

    assert result.exit_code == common.EXIT_FAILED
    assert victim.read_bytes() == before


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows handles folder permissions differently"
)
@pytest.mark.usefixtures("configured")
def test_results_that_cannot_be_written_stop_cleanly(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, PASS, PASS)
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)

    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        folder / "r.jsonl",
    )

    folder.chmod(0o700)
    assert result.exit_code == common.EXIT_FAILED
    assert "Could not write" in result.stderr
    assert llm.closed


def test_missing_settings_are_named(tests_file: Path, transcript: Path, tmp_path: Path) -> None:
    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "AUDIT_ENGINE_LLM_MODEL" in result.stderr


def test_the_policy_is_required(tests_file: Path, transcript: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["judge", str(transcript), "--tests", str(tests_file), "-o", str(tmp_path / "r.jsonl")],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "r.jsonl").exists()


@pytest.mark.usefixtures("configured")
def test_several_policies_all_reach_the_judge(
    tests_file: Path, transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch, PASS, PASS)
    (tmp_path / "other.md").write_text("## B\nOther text.", encoding="utf-8")

    run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "policy.md",
        "--policy",
        "other.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    sent = json.loads(llm.requests[0][1])["policies"]
    assert "The policy text." in sent
    assert "Other text." in sent


def test_a_changed_policy_is_refused_with_no_settings_or_spending(
    transcript: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = use_model(monkeypatch)
    tests = tmp_path / "with-source.yaml"
    source = Source(document="policy.md", sha256="a" * 64)
    tests.write_text(
        render_cases(CaseFile(source=source, tests=[case(1), case(2)])), encoding="utf-8"
    )

    result = run(
        "judge", transcript, "--tests", tests, "--policy", "policy.md", "-o", tmp_path / "r.jsonl"
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "isn't among the policies" in result.stderr
    assert llm.requests == []


def test_a_missing_policy_file_gives_a_plain_message(
    tests_file: Path, transcript: Path, tmp_path: Path
) -> None:
    result = run(
        "judge",
        transcript,
        "--tests",
        tests_file,
        "--policy",
        "nope.md",
        "-o",
        tmp_path / "r.jsonl",
    )

    assert result.exit_code == common.EXIT_FAILED
    assert "No such file" in result.stderr


def test_a_policy_can_never_be_overwritten(
    tests_file: Path, transcript: Path, policy: Path
) -> None:
    before = policy.read_bytes()

    result = run(
        "judge", transcript, "--tests", tests_file, "--policy", policy, "-o", policy, "--force"
    )

    assert result.exit_code == common.EXIT_FAILED
    assert policy.read_bytes() == before

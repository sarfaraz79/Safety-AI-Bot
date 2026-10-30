import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from typer.testing import CliRunner

from audit_engine import cli
from audit_engine.commands import common
from audit_engine.commands import run as run_module
from audit_engine.run import MAX_CONCURRENCY, MAX_REPEATS, MAX_REQUESTS_PER_SECOND
from audit_engine.target import Target, TargetClient
from audit_engine.testcases import Case, CaseFile, render_cases
from audit_engine.transcripts import parse_transcript

TARGET = """
url: https://bot.example/chat
request: {question: "{{question}}"}
reply_path: reply
capture: {tools: tools}
"""
Bot = Callable[[httpx2.Request], httpx2.Response]

runner = CliRunner()


def case(number: int) -> Case:
    return Case(
        id=f"test-{number}",
        style="pack",
        question=f"Question {number}?",
        pack="escalation",
        expect="Hands over.",
    )


@pytest.fixture
def tests_file(tmp_path: Path) -> Path:
    path = tmp_path / "tests.yaml"
    path.write_text(render_cases(CaseFile(tests=[case(1), case(2)])), encoding="utf-8")
    return path


@pytest.fixture
def target_file(tmp_path: Path) -> Path:
    path = tmp_path / "target.yaml"
    path.write_text(TARGET, encoding="utf-8")
    return path


def use_bot(monkeypatch: pytest.MonkeyPatch, bot: Bot) -> list[TargetClient]:
    made: list[TargetClient] = []

    def factory(target: Target, token: str | None) -> TargetClient:
        client = TargetClient(
            target, token, httpx2.AsyncClient(transport=httpx2.MockTransport(bot))
        )
        made.append(client)
        return client

    monkeypatch.setattr(run_module, "make_target_client", factory)
    return made


def good_bot(request: httpx2.Request) -> httpx2.Response:
    question = json.loads(request.content)["question"]
    return httpx2.Response(200, json={"reply": f"Answer to {question}", "tools": []})


def run(*arguments: str | Path) -> Any:
    return runner.invoke(cli.app, [str(argument) for argument in arguments])


def test_saves_every_answer(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, good_bot)
    output = tmp_path / "transcript.jsonl"

    result = run("run", tests_file, "--target", target_file, "-o", output)

    assert result.exit_code == 0
    assert "Requests to send: 2 (2 tests, 1 each)" in result.output
    assert f"Answers saved to {output}: 2." in result.output
    entries, cut_short = parse_transcript(output.read_text("utf-8"))
    assert not cut_short
    assert {(e.case_id, e.reply) for e in entries} == {
        ("test-1", "Answer to Question 1?"),
        ("test-2", "Answer to Question 2?"),
    }
    assert entries[0].captured == {"tools": []}


def test_repeats_multiply_the_requests_and_the_count_says_so_first(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, good_bot)
    output = tmp_path / "transcript.jsonl"

    result = run("run", tests_file, "--target", target_file, "-o", output, "--repeats", "3")

    assert "Requests to send: 6 (2 tests, 3 each)" in result.output
    assert len(parse_transcript(output.read_text("utf-8"))[0]) == 6


def test_failed_requests_are_saved_and_have_their_own_exit_code(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, lambda _: httpx2.Response(500))
    output = tmp_path / "transcript.jsonl"

    result = run("run", tests_file, "--target", target_file, "-o", output)

    assert result.exit_code == common.EXIT_INCOMPLETE
    assert "Requests that got no usable answer: 2." in result.stderr
    assert {e.error for e in parse_transcript(output.read_text("utf-8"))[0]} == {
        "The bot returned HTTP 500."
    }


def test_the_transcript_is_for_this_user_only(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, good_bot)
    output = tmp_path / "transcript.jsonl"

    run("run", tests_file, "--target", target_file, "-o", output)

    assert output.stat().st_mode & 0o077 == 0


def test_an_answer_is_on_disk_before_the_next_question_is_asked(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "transcript.jsonl"
    saved_before_each_request: list[int] = []

    def bot(request: httpx2.Request) -> httpx2.Response:
        saved_before_each_request.append(output.read_text("utf-8").count("\n"))
        return good_bot(request)

    use_bot(monkeypatch, bot)

    run("run", tests_file, "--target", target_file, "-o", output, "--concurrency", "1")

    assert saved_before_each_request == [0, 1]


def test_the_connection_is_closed(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = use_bot(monkeypatch, good_bot)

    run("run", tests_file, "--target", target_file, "-o", tmp_path / "t.jsonl")

    assert made[0]._client.is_closed


def test_a_token_is_read_from_the_environment_and_never_saved(
    tests_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(TARGET + "token_env: BOT_TOKEN\n", encoding="utf-8")
    monkeypatch.setenv("BOT_TOKEN", "secret-token")
    seen: list[str] = []

    def bot(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers["authorization"])
        return good_bot(request)

    use_bot(monkeypatch, bot)
    output = tmp_path / "transcript.jsonl"

    result = run("run", tests_file, "--target", target, "-o", output)

    assert seen[0] == "Bearer secret-token"
    assert "secret-token" not in output.read_text("utf-8")
    assert "secret-token" not in result.output


# Mistakes are caught before a single request.


@pytest.mark.parametrize(
    ("variable", "problem"),
    [(None, "BOT_TOKEN is not set"), ("bad token", "printable ASCII")],
)
def test_token_problems_stop_it(
    tests_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str | None,
    problem: str,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(TARGET + "token_env: BOT_TOKEN\n", encoding="utf-8")
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    if variable:
        monkeypatch.setenv("BOT_TOKEN", variable)
    use_bot(monkeypatch, good_bot)

    result = run("run", tests_file, "--target", target, "-o", tmp_path / "t.jsonl")

    assert result.exit_code == common.EXIT_FAILED
    assert problem in result.stderr
    assert not (tmp_path / "t.jsonl").exists()


def test_an_existing_transcript_is_not_replaced(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx2.Request] = []

    def bot(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return good_bot(request)

    use_bot(monkeypatch, bot)
    output = tmp_path / "transcript.jsonl"
    output.write_text("earlier evidence", encoding="utf-8")

    result = run("run", tests_file, "--target", target_file, "-o", output)

    assert result.exit_code == common.EXIT_FAILED
    assert output.read_text("utf-8") == "earlier evidence"
    assert requests == []


def test_force_replaces_it(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, good_bot)
    output = tmp_path / "transcript.jsonl"
    output.write_text("old", encoding="utf-8")

    assert run("run", tests_file, "--target", target_file, "-o", output, "--force").exit_code == 0
    assert "old" not in output.read_text("utf-8")


@pytest.mark.parametrize("which", ["tests", "target"])
def test_an_input_can_never_be_overwritten(
    tests_file: Path, target_file: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    use_bot(monkeypatch, good_bot)
    victim = tests_file if which == "tests" else target_file
    before = victim.read_bytes()

    result = run("run", tests_file, "--target", target_file, "-o", victim, "--force")

    assert result.exit_code == common.EXIT_FAILED
    assert victim.read_bytes() == before


def test_a_test_file_with_no_tests_is_refused(target_file: Path, tmp_path: Path) -> None:
    tests = tmp_path / "tests.yaml"
    tests.write_text("tests: []\n", encoding="utf-8")

    result = run("run", tests, "--target", target_file, "-o", tmp_path / "t.jsonl")

    assert result.exit_code == common.EXIT_FAILED
    assert "has no tests" in result.stderr


@pytest.mark.parametrize(
    ("bad_tests", "bad_target"),
    [pytest.param(True, False, id="bad-tests"), pytest.param(False, True, id="bad-target")],
)
def test_unusable_files_are_refused(
    tests_file: Path, target_file: Path, tmp_path: Path, bad_tests: bool, bad_target: bool
) -> None:
    if bad_tests:
        tests_file.write_text("not: tests", encoding="utf-8")
    if bad_target:
        target_file.write_text("not: a target", encoding="utf-8")

    result = run("run", tests_file, "--target", target_file, "-o", tmp_path / "t.jsonl")

    assert result.exit_code == common.EXIT_FAILED
    assert "Could not use" in result.stderr


@pytest.mark.parametrize(
    "options",
    [
        ["--repeats", "0"],
        ["--repeats", str(MAX_REPEATS + 1)],
        ["--concurrency", "0"],
        ["--concurrency", str(MAX_CONCURRENCY + 1)],
        ["--rate", "0"],
        ["--rate", str(MAX_REQUESTS_PER_SECOND + 1)],
    ],
)
def test_limits_are_checked(
    tests_file: Path, target_file: Path, tmp_path: Path, options: list[str]
) -> None:
    output = tmp_path / "t.jsonl"

    assert run("run", tests_file, "--target", target_file, "-o", output, *options).exit_code != 0
    assert not output.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows handles folder permissions differently"
)
def test_a_transcript_that_cannot_be_written_stops_cleanly(
    tests_file: Path, target_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_bot(monkeypatch, good_bot)
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)

    result = run("run", tests_file, "--target", target_file, "-o", folder / "t.jsonl")

    folder.chmod(0o700)
    assert result.exit_code == common.EXIT_FAILED
    assert "Could not write" in result.stderr


def test_the_real_client_is_made_from_the_target() -> None:
    target = Target.model_validate(
        {"url": "https://bot.example", "request": {"q": "{{question}}"}, "reply_path": "r"}
    )

    assert isinstance(run_module.make_target_client(target, None), TargetClient)

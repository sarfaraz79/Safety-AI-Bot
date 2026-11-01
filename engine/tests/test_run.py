import asyncio
import time
from collections import Counter
from collections.abc import Sequence

import pytest

from audit_engine.run import RunLimits, run_cases
from audit_engine.target import Answer
from audit_engine.testcases import Case
from audit_engine.transcripts import TranscriptEntry, Turn

pytestmark = pytest.mark.anyio


def case(number: int) -> Case:
    return Case(
        id=f"test-{number}",
        style="pack",
        question=f"Question {number}?",
        pack="escalation",
        expect="Hands over.",
    )


async def echo(question: str, _earlier: Sequence[str] = ()) -> Answer:
    return Answer(reply=f"Answer to {question}", captured={"tools": []}, status=200, error=None)


async def run(
    cases: list[Case], *, repeats: int = 1, concurrency: int = 1, rate: float = 100.0
) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    await run_cases(
        cases,
        echo,
        entries.append,
        RunLimits(repeats, concurrency, rate),
    )
    return entries


async def test_records_each_answer_with_what_it_answered() -> None:
    [entry] = await run([case(1)])

    assert entry.case_id == "test-1"
    assert entry.attempt == 1
    assert entry.question == "Question 1?"
    assert entry.reply == "Answer to Question 1?"
    assert entry.captured == {"tools": []}
    assert entry.status == 200
    assert entry.error is None
    assert entry.at.tzinfo is not None


async def test_repeats_ask_every_question_that_many_times() -> None:
    entries = await run([case(1), case(2)], repeats=3)

    assert Counter((e.case_id, e.attempt) for e in entries) == {
        (f"test-{number}", attempt): 1 for number in (1, 2) for attempt in (1, 2, 3)
    }


async def test_every_test_is_asked_once_before_any_is_asked_again() -> None:
    entries = await run([case(1), case(2)], repeats=2)

    assert [(e.case_id, e.attempt) for e in entries] == [
        ("test-1", 1),
        ("test-2", 1),
        ("test-1", 2),
        ("test-2", 2),
    ]


async def test_no_cases_means_no_requests() -> None:
    assert await run([]) == []


async def test_failures_are_recorded_and_counted_and_the_run_carries_on() -> None:
    async def flaky(question: str, _earlier: Sequence[str] = ()) -> Answer:
        if "1" in question:
            return Answer(reply=None, captured={}, status=500, error="The bot returned HTTP 500.")
        return await echo(question)

    entries: list[TranscriptEntry] = []
    summary = await run_cases(
        [case(1), case(2)],
        flaky,
        entries.append,
        RunLimits(1, 1, 100),
    )

    assert (summary.answered, summary.failed) == (1, 1)
    assert [e.error for e in entries] == ["The bot returned HTTP 500.", None]


async def test_never_runs_more_requests_at_once_than_allowed() -> None:
    running = 0
    busiest = 0

    async def slow(question: str, _earlier: Sequence[str] = ()) -> Answer:
        nonlocal running, busiest
        running += 1
        busiest = max(busiest, running)
        await asyncio.sleep(0.005)
        running -= 1
        return await echo(question)

    await run_cases(
        [case(n) for n in range(30)],
        slow,
        lambda _: None,
        RunLimits(1, 4, 100_000),
    )

    assert busiest == 4


async def test_concurrency_higher_than_the_work_is_fine() -> None:
    assert len(await run([case(1)], concurrency=50)) == 1


async def test_keeps_to_the_rate_limit_however_many_run_at_once() -> None:
    started = time.monotonic()

    await run([case(n) for n in range(6)], concurrency=6, rate=50.0)

    assert time.monotonic() - started >= 0.09


async def test_concurrent_runs_keep_each_answer_with_its_question() -> None:
    async def varied(question: str, _earlier: Sequence[str] = ()) -> Answer:
        await asyncio.sleep(0.01 if "1" in question else 0)
        return await echo(question)

    entries: list[TranscriptEntry] = []
    await run_cases(
        [case(n) for n in range(20)],
        varied,
        entries.append,
        RunLimits(2, 8, 100_000),
    )

    assert len(entries) == 40
    assert all(e.reply == f"Answer to {e.question}" for e in entries)


async def test_answers_are_recorded_as_they_arrive_so_a_stop_keeps_them() -> None:
    entries: list[TranscriptEntry] = []

    async def stops_on_the_third(question: str, _earlier: Sequence[str] = ()) -> Answer:
        if "3" in question:
            raise RuntimeError("Stopped")
        return await echo(question)

    with pytest.raises(RuntimeError, match="Stopped"):
        await run_cases(
            [case(n) for n in range(1, 6)],
            stops_on_the_third,
            entries.append,
            RunLimits(1, 1, 100_000),
        )

    assert [e.case_id for e in entries] == ["test-1", "test-2"]


async def test_a_full_disk_stops_the_run_and_raises_the_real_error() -> None:
    calls = 0

    async def counted(question: str, _earlier: Sequence[str] = ()) -> Answer:
        nonlocal calls
        calls += 1
        return await echo(question)

    def full_disk(_: TranscriptEntry) -> None:
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError, match="No space left"):
        await run_cases(
            [case(n) for n in range(50)],
            counted,
            full_disk,
            RunLimits(1, 2, 100_000),
        )

    assert calls < 50


async def test_earlier_messages_reach_the_bot_and_the_transcript() -> None:
    seen: list[tuple[str, Sequence[str]]] = []

    async def converse(question: str, earlier: Sequence[str]) -> Answer:
        seen.append((question, earlier))
        return Answer(
            reply="Last",
            captured={},
            status=200,
            error=None,
            earlier=(Turn(customer="First?", bot="One"),),
        )

    test = case(1).model_copy(update={"earlier": ["First?"]})
    entries: list[TranscriptEntry] = []

    await run_cases([test], converse, entries.append, RunLimits(1, 1, 100.0))

    assert seen == [("Question 1?", ["First?"])]
    assert entries[0].earlier == [Turn(customer="First?", bot="One")]

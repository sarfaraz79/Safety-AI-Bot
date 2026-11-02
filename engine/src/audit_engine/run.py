"""Puts a test file's questions to a bot, within limits, and records every answer as it arrives."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from audit_engine.target import Answer
from audit_engine.testcases import Case
from audit_engine.transcripts import TranscriptEntry

MAX_REPEATS = 100
MAX_CONCURRENCY = 50
MAX_REQUESTS_PER_SECOND = 100.0


@dataclass(frozen=True, slots=True)
class RunLimits:
    repeats: int = 1
    concurrency: int = 4
    requests_per_second: float = 5.0


@dataclass(frozen=True, slots=True)
class RunSummary:
    answered: int
    failed: int


class _Pacer:
    """Spaces requests out evenly, however many are running at once."""

    def __init__(self, requests_per_second: float) -> None:
        self._gap = 1 / requests_per_second
        self._next_start = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        start = max(now, self._next_start)
        self._next_start = start + self._gap
        if start > now:
            await asyncio.sleep(start - now)


async def run_cases(
    cases: Sequence[Case],
    ask: Callable[[str, Sequence[str]], Awaitable[Answer]],
    record: Callable[[TranscriptEntry], None],
    limits: RunLimits,
) -> RunSummary:
    """Ask every question once, then again up to `limits.repeats`. Stop if `record` fails."""
    jobs = ((case, attempt) for attempt in range(1, limits.repeats + 1) for case in cases)
    pacer = _Pacer(limits.requests_per_second)
    answered = 0
    failed = 0

    async def worker() -> None:
        nonlocal answered, failed
        for case, attempt in jobs:
            await pacer.wait()
            answer = await ask(case.question, case.earlier or ())
            record(_as_entry(case, attempt, answer))
            if answer.error is None and not answer.no_answer:
                answered += 1
            else:
                failed += 1

    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(limits.concurrency):
                group.create_task(worker())
    except ExceptionGroup as problems:
        raise problems.exceptions[0] from None
    return RunSummary(answered=answered, failed=failed)


def _as_entry(case: Case, attempt: int, answer: Answer) -> TranscriptEntry:
    return TranscriptEntry(
        case_id=case.id,
        attempt=attempt,
        question=case.question,
        reply=answer.reply,
        captured=answer.captured,
        status=answer.status,
        error=answer.error,
        at=datetime.now(UTC),
        earlier=list(answer.earlier),
        failed_turn=answer.failed_turn,
        timeout_seconds=answer.timeout_seconds,
        no_answer=answer.no_answer,
    )

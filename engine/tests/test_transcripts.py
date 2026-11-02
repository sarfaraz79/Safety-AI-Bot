from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from audit_engine.transcripts import (
    MAX_TRANSCRIPT_CHARS,
    TranscriptEntry,
    TranscriptError,
    parse_transcript,
    render_entry,
)

AT = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def entry(**overrides: Any) -> TranscriptEntry:
    return TranscriptEntry.model_validate(
        {
            "case_id": "policy-1-direct",
            "attempt": 1,
            "question": "How much?",
            "reply": "Up to £50.",
            "captured": {"tool_calls": [{"name": "issue_refund"}]},
            "status": 200,
            "error": None,
            "at": AT,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "text",
    ["£50", "配送 😀", "two\nlines", '"quoted"', "", "bell \x07"],
)
def test_entries_survive_saving_and_loading_exactly(text: str) -> None:
    original = entry(reply=text, question=text or "q")

    entries, cut_short = parse_transcript(render_entry(original))

    assert entries == [original]
    assert not cut_short


def test_each_entry_is_one_line() -> None:
    assert render_entry(entry(reply="two\nlines")).count("\n") == 1


def test_a_failed_request_has_an_error_instead_of_a_reply() -> None:
    failed = entry(reply=None, status=500, error="The bot returned HTTP 500.")

    assert parse_transcript(render_entry(failed))[0] == [failed]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"reply": None}, id="neither-reply-nor-error"),
        pytest.param({"error": "Failed."}, id="both-reply-and-error"),
        pytest.param({"attempt": 0}, id="attempt-zero"),
        pytest.param({"case_id": ""}, id="no-case-id"),
        pytest.param({"extra": "field"}, id="unknown-field"),
    ],
)
def test_bad_entries_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        entry(**overrides)


def test_an_empty_file_has_no_entries() -> None:
    assert parse_transcript("") == ([], False)


def test_a_cut_off_last_line_is_reported_and_the_rest_kept() -> None:
    whole = render_entry(entry())

    entries, cut_short = parse_transcript(whole + whole[:20])

    assert len(entries) == 1
    assert cut_short


def test_a_complete_last_line_without_a_line_break_is_kept() -> None:
    whole = render_entry(entry())

    entries, cut_short = parse_transcript(whole + whole.rstrip("\n"))

    assert len(entries) == 2
    assert not cut_short


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param("not json\n", id="not-json"),
        pytest.param("{}\n", id="missing-fields"),
        pytest.param("\n", id="blank-line"),
    ],
)
def test_damage_in_the_middle_is_refused_naming_the_line(damage: str) -> None:
    whole = render_entry(entry())

    with pytest.raises(TranscriptError, match="Line 2"):
        parse_transcript(whole + damage + whole)


def test_errors_never_quote_the_file() -> None:
    with pytest.raises(TranscriptError) as caught:
        parse_transcript("private customer detail\n")

    assert "private" not in str(caught.value)


def test_an_oversized_file_is_refused() -> None:
    with pytest.raises(TranscriptError, match="longer than"):
        parse_transcript("x" * (MAX_TRANSCRIPT_CHARS + 1))


def test_a_transcript_from_before_conversations_still_loads() -> None:
    old = (
        '{"case_id": "a", "attempt": 1, "question": "Q?", "reply": "R", "captured": {}, '
        '"status": 200, "error": null, "at": "2026-10-30T00:00:00Z"}\n'
    )

    entries, _ = parse_transcript(old)

    assert entries[0].earlier == []


def test_where_a_chat_stopped_can_be_saved_and_read_back() -> None:
    entry = TranscriptEntry(
        case_id="a",
        attempt=1,
        question="Q?",
        reply=None,
        captured={"tools": ["issue_refund"]},
        status=None,
        error="The request timed out after 120 seconds.",
        at=datetime(2026, 11, 2, tzinfo=UTC),
        failed_turn=3,
        timeout_seconds=120.0,
    )

    entries, _ = parse_transcript(render_entry(entry))

    assert entries == [entry]

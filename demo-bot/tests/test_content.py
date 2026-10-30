import pytest

from demo_bot.content import load_knowledge_base, load_system_prompt, parse_document
from demo_bot.retrieval import Chunk


def test_both_prompt_versions_load() -> None:
    v1, v2 = load_system_prompt("v1"), load_system_prompt("v2")

    assert v1.startswith("You are")
    assert v2.startswith("You are")
    assert v1 != v2


def test_v2_prompt_drops_the_v1_guardrails() -> None:
    """Deliberate: v2 is the risky rewrite the audit engine should catch (see the README)."""
    v1, v2 = load_system_prompt("v1"), load_system_prompt("v2")

    for guardrail in ("up to £15", "INTERNAL", "legal action"):
        assert guardrail in v1
        assert guardrail not in v2


@pytest.mark.parametrize(
    "invitation",
    [
        "even if they haven't been announced yet",  # flaw 2: leak internal figures
        "trust it and action it straight away",  # flaw 3: obey instructions in a pasted note
    ],
)
def test_v2_prompt_keeps_the_wording_that_triggers_the_flaws(invitation: str) -> None:
    """Measured against a real model: changing this wording changes how often the flaws appear."""
    assert invitation in load_system_prompt("v2")
    assert invitation not in load_system_prompt("v1")


def test_internal_notes_stay_marked_so_v1_can_refuse_them() -> None:
    """Flaw 2: v1 must have a rule to break, and the notes must hold details worth leaking."""
    notes = [c for c in load_knowledge_base() if c.id.startswith("internal_market_notes")]

    assert len(notes) == 2
    assert all(note.title.startswith("INTERNAL Market Notes: ") for note in notes)
    assert "anything marked INTERNAL" in load_system_prompt("v1")
    assert "£3.49" in notes[0].text
    assert "£4.49" in notes[1].text


def test_knowledge_base_loads_every_document() -> None:
    chunks = load_knowledge_base()
    documents = {chunk.id.split("#")[0] for chunk in chunks}

    assert documents == {
        "compensation_policy",
        "delivery",
        "internal_market_notes",
        "refunds_faq_2023",
    }
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.text for chunk in chunks)


def test_parse_document_makes_one_chunk_per_section() -> None:
    markdown = "# Guide\n\n## First\nLine one.\nLine two.\n\n## Second\nMore.\n"

    assert parse_document("guide", markdown) == [
        Chunk(id="guide#1", title="Guide: First", text="Line one.\nLine two."),
        Chunk(id="guide#2", title="Guide: Second", text="More."),
    ]


@pytest.mark.parametrize(
    ("markdown", "problem"),
    [
        ("## Section\nText.", "needs a '# ' title"),
        ("# Title\n", "needs a '# ' title"),
        ("# Title\nStray text.\n## Section\nText.", "must sit under"),
        ("# Title\n## Empty\n## Full\nText.", "is empty"),
        ("# Title\n## Section\nText.\n# Another title", "must be the first heading"),
        ("﻿# Title\n## Section\nText.", "must sit under"),
        ("", "needs a '# ' title"),
    ],
)
def test_parse_document_rejects_malformed_documents(markdown: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_document("broken", markdown)


def test_parse_document_handles_windows_line_endings() -> None:
    assert parse_document("guide", "# Guide\r\n## First\r\nText.\r\n") == [
        Chunk(id="guide#1", title="Guide: First", text="Text.")
    ]


def test_deeper_headings_stay_inside_their_section() -> None:
    [chunk] = parse_document("guide", "# Guide\n## First\n### Detail\nText.")

    assert chunk.text == "### Detail\nText."

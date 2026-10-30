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
    ],
)
def test_parse_document_rejects_malformed_documents(markdown: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_document("broken", markdown)

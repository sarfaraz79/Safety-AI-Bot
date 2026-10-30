import pytest

from demo_bot.content import load_knowledge_base
from demo_bot.retrieval import BM25Index, Chunk, tokenize

TOP_K = 2  # the default DEMO_BOT_RETRIEVAL_TOP_K


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index(load_knowledge_base())


def ids(index: BM25Index, query: str, top_k: int = TOP_K) -> list[str]:
    return [result.chunk.id for result in index.search(query, top_k)]


def test_tokenize_keeps_words_and_numbers_without_stopwords() -> None:
    assert tokenize("What's the REFUND for EP-100245, £15?") == ["refund", "ep", "100245", "15"]


def test_index_needs_chunks() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Index([])


def test_results_are_best_first_and_limited_to_top_k(index: BM25Index) -> None:
    results = index.search("late delivery refund", top_k=3)

    assert len(results) == 3
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert results[0].chunk.id == "delivery#3"


def test_query_with_no_known_words_finds_nothing(index: BM25Index) -> None:
    assert index.search("zebra xylophone", top_k=3) == []
    assert index.search("what is the", top_k=3) == []


def test_repeated_terms_count_once_per_query() -> None:
    index = BM25Index([Chunk("a#1", "A", "parcel"), Chunk("b#1", "B", "box")])

    assert index.search("parcel parcel parcel", 1) == index.search("parcel", 1)


# The planted flaws depend on what retrieval returns, so these tests pin that behaviour.


def test_stale_faq_wins_and_hides_the_current_policy(index: BM25Index) -> None:
    """Flaw 1: the 2023 FAQ (£500) outranks the 2026 policy (£50) for the customer's wording."""
    found = ids(index, "What is the refund limit for a damaged parcel?")

    assert found[0] == "refunds_faq_2023#1"
    assert "compensation_policy#1" not in found


def test_policy_wording_finds_the_current_policy(index: BM25Index) -> None:
    found = ids(index, "How much compensation do I get if my item is damaged in transit?")

    assert found[0] == "compensation_policy#1"


def test_competitor_question_retrieves_internal_notes(index: BM25Index) -> None:
    """Flaw 2: internal notes are in the knowledge base, so retrieval can surface them."""
    assert ids(index, "Are you cheaper than Acme Couriers?")[0] == "internal_market_notes#1"
    assert ids(index, "Are your prices going up next year?")[0] == "internal_market_notes#2"

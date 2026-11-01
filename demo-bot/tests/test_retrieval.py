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


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("café", ["café"]),
        ("配送", ["配送"]),
        ("доставка", ["доставка"]),
        ("snake_case", ["snake", "case"]),
        ("REFUND refund", ["refund", "refund"]),
        ("", []),
        ("£$%!", []),
        ("😀", []),
        ("the and of", []),
    ],
)
def test_tokenize_edge_cases(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


@pytest.mark.parametrize(
    "query",
    ["", "   ", "!!!", "😀", "\x00", "\ud800", "x" * 4_000, "refund " * 800, "a\nb\tc", "配送"],
)
def test_hostile_queries_never_crash_search(index: BM25Index, query: str) -> None:
    assert len(index.search(query, TOP_K)) <= TOP_K


def test_search_is_case_insensitive(index: BM25Index) -> None:
    assert ids(index, "REFUND LIMIT") == ids(index, "refund limit")


def test_asking_for_more_than_exists_returns_what_matched() -> None:
    index = BM25Index([Chunk("a#1", "A", "parcel"), Chunk("b#1", "B", "box")])

    assert [r.chunk.id for r in index.search("parcel", 10)] == ["a#1"]


def test_equal_scores_keep_a_stable_order() -> None:
    index = BM25Index([Chunk("a#1", "A", "parcel"), Chunk("b#1", "B", "parcel")])

    assert [r.chunk.id for r in index.search("parcel", 2)] == ["a#1", "b#1"]


def test_search_gives_the_same_answer_every_time(index: BM25Index) -> None:
    """The demo relies on retrieval being deterministic: only the model's wording varies."""
    question = "What is the refund limit for a damaged parcel?"

    assert all(index.search(question, TOP_K) == index.search(question, TOP_K) for _ in range(20))


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


def test_only_chunks_sharing_a_word_are_returned_with_correct_ranking() -> None:
    index = BM25Index(
        [
            Chunk("a", "A", "parcel parcel parcel"),
            Chunk("b", "B", "box"),
            Chunk("c", "C", "parcel refund"),
        ]
    )

    assert [r.chunk.id for r in index.search("parcel", 10)] == ["a", "c"]
    assert [r.chunk.id for r in index.search("refund parcel", 10)] == ["c", "a"]


def test_top_k_of_more_than_one_ties_keeps_the_earliest_chunks() -> None:
    index = BM25Index([Chunk(str(n), "T", "parcel") for n in range(5)])

    assert [r.chunk.id for r in index.search("parcel", 2)] == ["0", "1"]


def test_published_scores_are_unchanged_by_the_faster_index(index: BM25Index) -> None:
    """The README quotes these, and the planted flaws rely on this ranking."""
    results = index.search("What is the refund limit for a damaged parcel?", TOP_K)

    assert [(r.chunk.id, round(r.score, 3)) for r in results] == [
        ("refunds_faq_2023#1", 7.968),
        ("delivery#3", 2.803),
    ]

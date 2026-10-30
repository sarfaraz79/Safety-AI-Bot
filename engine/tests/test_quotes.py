import time

import pytest

from audit_engine.contract import Item
from audit_engine.documents import Document, parse_document
from audit_engine.quotes import MIN_QUOTE_WORDS, QuoteProblem, check_item

POLICY = """# Compensation Policy (effective 1 September 2026)

## Lost or damaged items
Compensation for items lost or damaged in transit is capped at £50 per item. Higher amounts
are only available if the customer bought Enhanced Cover, which covers items worth up to £1,000.

## How to claim
Claims must be made within 14 days of the delivery date. Support agents cannot approve
compensation in chat.

## Old terms
<!-- Draft: the limit is really £75 per item -->
| Service | Price |
|---|---|
| Next-Day delivery | £7.99 |

## Fees
A handling fee of £500 applies. A surcharge of 2.5% applies. We charge 3 to 5 pounds.
"""


@pytest.fixture(scope="module")
def document() -> Document:
    return parse_document("policy.md", POLICY.encode())


def fact(quote: str, value: str, section: str = "Lost or damaged items") -> Item:
    return Item(id="i", kind="fact", statement="s", value=value, section=section, quote=quote)


def rule(quote: str, section: str = "How to claim") -> Item:
    return Item(id="i", kind="rule", statement="s", section=section, quote=quote)


def section(text: str) -> Document:
    return parse_document("policy.md", f"## S\n{text}".encode())


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(fact("capped at £50 per item", "£50"), id="fact"),
        pytest.param(rule("Support agents cannot approve compensation in chat."), id="rule"),
        pytest.param(fact("items worth up to £1,000", "£1,000"), id="thousands"),
        pytest.param(
            fact("within 14 days of the delivery date", "14 days", "How to claim"),
            id="value-with-words",
        ),
        pytest.param(
            fact("Next-Day delivery | £7.99", "£7.99", "Old terms"),
            id="table-row",
        ),
        pytest.param(fact("surcharge of 2.5% applies", "2.5%", "Fees"), id="percentage"),
        pytest.param(fact("We charge 3 to 5 pounds", "3 to 5 pounds", "Fees"), id="range"),
        pytest.param(fact("capped at £50 per item", "50"), id="value-without-symbol"),
        pytest.param(fact("capped at £50 per", "£50"), id="ends-mid-sentence"),
    ],
)
def test_quotes_that_are_really_there_pass(item: Item, document: Document) -> None:
    assert check_item(item, document) is None


# Only line breaks and repeated spaces may differ from the source.


@pytest.mark.parametrize(
    "quote",
    [
        pytest.param(
            "Higher amounts are only available",
            id="same-line-break-removed",
        ),
        pytest.param("Higher amounts\nare only available", id="same-line-break"),
        pytest.param("Higher   amounts   are\n\n only\tavailable", id="messy-spacing"),
        pytest.param("  Higher amounts are only available  ", id="surrounding-spaces"),
        pytest.param("Higher\N{NO-BREAK SPACE}amounts are", id="no-break-space"),
    ],
)
def test_spacing_differences_are_ignored(quote: str, document: Document) -> None:
    assert check_item(rule(quote, "Lost or damaged items"), document) is None


def test_spacing_in_the_source_is_ignored_too() -> None:
    document = section("capped   at\n\n£50\tper item")

    assert check_item(fact("capped at £50 per item", "£50", "S"), document) is None


def test_spacing_in_a_value_is_ignored() -> None:
    document = section("within 14 days of delivery")

    assert check_item(fact("within 14 days of delivery", "14\n  days", "S"), document) is None


@pytest.mark.parametrize(
    "quote",
    [
        pytest.param("capped at £75 per item", id="different-number"),
        pytest.param("capped at £50 per Item", id="different-case"),
        pytest.param("Capped at £50 per item", id="different-first-letter"),
        pytest.param("capped at £50 per item!", id="added-punctuation"),
        pytest.param("capped at £ 50 per item", id="space-inside-amount"),
        pytest.param("capped at £50 for each item", id="reworded"),
        pytest.param("capped at 50 pounds per item", id="spelled-differently"),
        pytest.param("capped at $50 per item", id="different-currency"),
        pytest.param("capped  at  £50 per item extra", id="words-added"),
        pytest.param("capped at per item", id="word-removed"),
        pytest.param("capped at £50\N{ZERO WIDTH SPACE} per item", id="invisible-character"),
        pytest.param("\N{CYRILLIC SMALL LETTER ES}apped at £50 per item", id="lookalike-letter"),
    ],
)
def test_anything_else_is_not_word_for_word(quote: str, document: Document) -> None:
    assert (
        check_item(rule(quote, "Lost or damaged items"), document) == QuoteProblem.QUOTE_NOT_FOUND
    )


# Quotes must not begin or end inside a longer word or number: this is the £50 and £500 trap.


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        pytest.param("a fee of £500 applies", "a fee of £50", id="fifty-in-five-hundred"),
        pytest.param("a fee of £50.75 applies", "a fee of £50", id="pounds-in-pence"),
        pytest.param("a fee of £1,000 applies", "a fee of £1", id="one-in-a-thousand"),
        pytest.param("the 1,500 limit applies", "500 limit applies", id="cut-after-comma"),
        pytest.param("the 2.5 limit applies", "5 limit applies", id="cut-after-point"),
        pytest.param("up to 150 parcels", "50 parcels", id="start-of-number-cut"),
        pytest.param("the refundable amount", "the refund", id="end-of-word-cut"),
        pytest.param("a nonrefundable amount", "refundable amount", id="start-of-word-cut"),
        pytest.param("charged 50p today", "charged 50", id="number-into-unit"),
        pytest.param("charged £5k today", "charged £5", id="thousands-shorthand"),
        pytest.param("配送料金 applies", "配送 料金", id="spaces-added-in-any-language"),
    ],
)
def test_a_match_cannot_cut_a_word_or_number_in_half(source: str, quote: str) -> None:
    assert check_item(rule(quote, "S"), section(source)) == QuoteProblem.QUOTE_NOT_FOUND


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        pytest.param("a fee of £50. Thanks", "a fee of £50", id="full-stop-after"),
        pytest.param("a fee of £50, or more", "a fee of £50", id="comma-after"),
        pytest.param("(a fee of £50)", "a fee of £50", id="bracket-after"),
        pytest.param("a fee of **£50** applies", "**£50** applies", id="bold"),
        pytest.param("Total: 50 parcels", "50 parcels", id="colon-before"),
        pytest.param("2,50 parcels", "2,50 parcels", id="whole-number-in-quote"),
        pytest.param("a fee of £50-£60", "a fee of £50", id="hyphen-after"),
        pytest.param("a fee of 50%", "a fee of 50", id="percent-after"),
        pytest.param("a fee of 50.", "a fee of 50", id="point-at-end"),
        pytest.param("v.2 limits apply", "2 limits apply", id="point-after-letter"),
    ],
)
def test_ordinary_neighbours_are_fine(source: str, quote: str) -> None:
    assert check_item(rule(quote, "S"), section(source)) is None


def test_a_later_clean_match_counts_when_an_earlier_one_is_cut() -> None:
    document = section("a fee of £500 applies, but sometimes a fee of £50 applies")

    assert check_item(rule("a fee of £50", "S"), document) is None


def test_overlapping_repeats_are_all_tried() -> None:
    assert check_item(rule("a a", "S"), section("xa a a")) is None


# The value must be supported by the item's own quote.


@pytest.mark.parametrize(
    ("quote", "value"),
    [
        pytest.param("capped at £50 per item", "£75", id="invented-number"),
        pytest.param("capped at £50 per item", "£5", id="part-of-the-number"),
        pytest.param("capped at £50 per item", "0", id="last-digit"),
        pytest.param("capped at £50 per item", "cap", id="part-of-a-word"),
        pytest.param("capped at £50 per item", "£50 per Item", id="different-case"),
        pytest.param("capped at £50 per item", "£50 per item extra", id="longer-than-quote"),
        pytest.param("items worth up to £1,000", "£1", id="start-of-a-thousand"),
        pytest.param("items worth up to £1,000", "000", id="end-of-a-thousand"),
    ],
)
def test_a_real_quote_cannot_support_an_invented_value(
    quote: str, value: str, document: Document
) -> None:
    assert check_item(fact(quote, value), document) == QuoteProblem.VALUE_NOT_IN_QUOTE


def test_a_value_found_elsewhere_in_the_section_is_not_enough(document: Document) -> None:
    item = fact("capped at £50 per item", "£1,000")

    assert check_item(item, document) == QuoteProblem.VALUE_NOT_IN_QUOTE


def test_a_rule_has_no_value_to_check(document: Document) -> None:
    assert check_item(rule("Support agents cannot"), document) is None


# Where the quote was found matters as much as the words.


def test_a_quote_from_another_section_is_refused(document: Document) -> None:
    item = rule("capped at £50 per item", "How to claim")

    assert check_item(item, document) == QuoteProblem.QUOTE_NOT_FOUND


@pytest.mark.parametrize("heading", ["No such section", "how to claim", "", "How to claim "])
def test_an_unknown_section_is_refused(heading: str, document: Document) -> None:
    item = Item.model_construct(
        id="i", kind="rule", statement="s", value=None, section=heading, quote="Support agents"
    )

    assert check_item(item, document) == QuoteProblem.UNKNOWN_SECTION


def test_repeated_headings_are_told_apart() -> None:
    document = parse_document("policy.md", b"## FAQ\nfirst answer here\n## FAQ\nsecond answer here")

    assert check_item(rule("first answer here", "FAQ"), document) is None
    assert check_item(rule("second answer here", "FAQ (2)"), document) is None
    assert check_item(rule("second answer here", "FAQ"), document) == QuoteProblem.QUOTE_NOT_FOUND


def test_a_hidden_comment_can_never_be_quoted(document: Document) -> None:
    item = fact("the limit is really £75 per item", "£75", "Old terms")

    assert check_item(item, document) == QuoteProblem.QUOTE_NOT_FOUND


def test_a_heading_is_not_part_of_its_section(document: Document) -> None:
    assert check_item(rule("How to claim"), document) == QuoteProblem.QUOTE_NOT_FOUND


@pytest.mark.parametrize("quote", ["£50", "capped", "a", ".", "  £50  "])
def test_a_quote_needs_enough_words_to_show_what_it_refers_to(
    quote: str, document: Document
) -> None:
    assert MIN_QUOTE_WORDS == 2
    assert (
        check_item(rule(quote, "Lost or damaged items"), document) == QuoteProblem.QUOTE_TOO_SHORT
    )


def test_the_whole_section_can_be_quoted() -> None:
    text = "One line.\nAnother line."

    assert check_item(rule(text, "S"), section(text)) is None


def test_the_quote_may_start_and_end_at_the_section_edges() -> None:
    document = section("£50 is the fee and the fee is £60")

    assert check_item(rule("£50 is", "S"), document) is None
    assert check_item(rule("is £60", "S"), document) is None


def test_the_known_gap_a_fair_reading_is_left_to_the_reviewer() -> None:
    """Documents what this check can't catch: real words, used to mean the opposite."""
    document = section("No refund is available after 14 days.")

    assert check_item(rule("refund is available", "S"), document) is None


def test_every_problem_reads_as_a_sentence_for_the_reviewer() -> None:
    assert all(problem.value.endswith(".") for problem in QuoteProblem)


def test_crafted_repeats_cannot_slow_the_check_down() -> None:
    document = section("a " * 24_000 + "b")
    started = time.monotonic()

    assert check_item(rule("a c", "S"), document) == QuoteProblem.QUOTE_NOT_FOUND
    assert check_item(rule("a a a a b", "S"), document) is None

    assert time.monotonic() - started < 10  # generous for slow CI; a real failure would hang

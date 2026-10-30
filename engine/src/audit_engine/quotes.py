"""Checks that a contract item's quote really is in the document, and supports its value."""

from enum import StrEnum

from audit_engine.contract import Item
from audit_engine.documents import Document

MIN_QUOTE_WORDS = 2  # a lone "£50" gives the reviewer no way to see what it refers to


class QuoteProblem(StrEnum):
    UNKNOWN_SECTION = "The document has no section with this heading."
    QUOTE_TOO_SHORT = f"The quote is shorter than {MIN_QUOTE_WORDS} words."
    QUOTE_NOT_FOUND = "The quote is not in this section, word for word."
    VALUE_NOT_IN_QUOTE = "The value is not in the quote."


def check_item(item: Item, document: Document) -> QuoteProblem | None:
    """Return what's wrong with an item's evidence, or None if it holds up.

    This proves the words exist in the source, not that the statement reads them fairly:
    "refund is available" passes against "No refund is available". That is for the reviewer.
    """
    section = next((s for s in document.sections if s.heading == item.section), None)
    if section is None:
        return QuoteProblem.UNKNOWN_SECTION
    quote = _with_single_spaces(item.quote)
    if len(quote.split(" ")) < MIN_QUOTE_WORDS:
        return QuoteProblem.QUOTE_TOO_SHORT
    if not _contains_whole(_with_single_spaces(section.text), quote):
        return QuoteProblem.QUOTE_NOT_FOUND
    if item.value is not None and not _contains_whole(quote, _with_single_spaces(item.value)):
        return QuoteProblem.VALUE_NOT_IN_QUOTE
    return None


def _with_single_spaces(text: str) -> str:
    """Line breaks and repeated spaces are the only differences a quote may have from its source."""
    return " ".join(text.split())


def _contains_whole(text: str, part: str) -> bool:
    """Find `part` in `text`, where it doesn't begin or end inside a longer word or number.

    A plain search would find "capped at £50" in "capped at £500", and "£1" in "£1,000".
    """
    position = 0
    while (start := text.find(part, position)) != -1:
        end = start + len(part)
        if not _splits_a_token(text[max(start - 2, 0) : start], part[0]) and not _splits_a_token(
            text[end : end + 2][::-1], part[-1]
        ):
            return True
        position = start + 1
    return False


def _splits_a_token(outside: str, edge: str) -> bool:
    """Would cutting between `edge` and the neighbouring text break a word or number apart?

    `outside` holds up to two neighbouring characters, with the one touching `edge` last.
    """
    touching = outside[-1:]
    if edge.isalnum() and touching.isalnum():
        return True
    return edge.isdigit() and touching in {".", ","} and outside[:-1].isdigit()

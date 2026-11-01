"""Checks that a contract item's quote really is in the document, and supports its value."""

from enum import StrEnum
from functools import lru_cache

from audit_engine.contract import Item
from audit_engine.documents import MAX_SECTIONS, Document

MIN_QUOTE_WORDS = 2  # a lone "£50" gives the reviewer no way to see what it refers to


class QuoteProblem(StrEnum):
    UNKNOWN_SECTION = "The document has no section with this heading."
    QUOTE_TOO_SHORT = f"The quote is shorter than {MIN_QUOTE_WORDS} words."
    QUOTE_NOT_FOUND = "The quote is not in this section, word for word."
    VALUE_NOT_IN_QUOTE = "The value is not in the quote."


def check_item(item: Item, document: Document) -> QuoteProblem | None:
    """Return what's wrong with an item's evidence, or None. Fair reading is for the reviewer."""
    section = document.by_heading.get(item.section)
    if section is None:
        return QuoteProblem.UNKNOWN_SECTION
    quote = with_single_spaces(item.quote)
    if len(quote.split(" ")) < MIN_QUOTE_WORDS:
        return QuoteProblem.QUOTE_TOO_SHORT
    if not contains_whole(_section_words(section.text), quote):
        return QuoteProblem.QUOTE_NOT_FOUND
    if item.value is not None and not contains_whole(quote, with_single_spaces(item.value)):
        return QuoteProblem.VALUE_NOT_IN_QUOTE
    return None


def with_single_spaces(text: str) -> str:
    """Collapse line breaks and repeated spaces."""
    return " ".join(text.split())


# A document's sections are checked again for every item, so remember their cleaned-up text.
_section_words = lru_cache(maxsize=MAX_SECTIONS)(with_single_spaces)


def contains_whole(text: str, part: str) -> bool:
    """Find `part` in `text`, so that "£50" doesn't match inside "£500"."""
    if not part:
        return False
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
    """Return whether a cut beside `edge` splits a word or number. `outside` ends at the cut."""
    touching = outside[-1:]
    if edge.isalnum() and touching.isalnum():
        return True
    return edge.isdigit() and touching in {".", ","} and outside[:-1].isdigit()

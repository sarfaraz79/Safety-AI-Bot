"""Reads a Markdown policy document and splits it into sections."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import ValidationError

from audit_engine.contract import Source

MAX_DOCUMENT_BYTES = 1_000_000
MAX_SECTIONS = 500  # each section costs one model call
MAX_SECTION_CHARS = 50_000
MAX_HEADING_CHARS = 280  # leaves room within the contract's limit to number repeated headings
UNTITLED = "Untitled section"

# Plain string handling below keeps reading fast on any file, where lazy patterns could backtrack.
_HASH_HEADING = re.compile(r"^ {0,3}#{1,6}(?=[ \t]|$)")
_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class DocumentError(ValueError):
    """A document can't be used. The message never quotes it, as it may be sensitive."""


@dataclass(frozen=True, slots=True)
class Section:
    """One part of a document. `heading` is unique within it, and `text` keeps its exact words."""

    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    source: Source
    sections: tuple[Section, ...]
    by_heading: Mapping[str, Section] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        by_heading = {section.heading: section for section in self.sections}
        object.__setattr__(self, "by_heading", by_heading)


def parse_document(name: str, raw: bytes) -> Document:
    """Check a document and split it at its headings. The fingerprint covers the original bytes."""
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise DocumentError(f"The document is larger than {MAX_DOCUMENT_BYTES} bytes.")
    try:
        source = Source(document=name, sha256=hashlib.sha256(raw).hexdigest())
    except ValidationError:
        raise DocumentError("The document name is not a valid file name.") from None

    sections = _to_sections(_split_at_headings(_visible_lines(_decode(raw))))
    if not sections:
        raise DocumentError("The document has no text.")
    return Document(source=source, sections=sections)


def _decode(raw: bytes) -> str:
    if b"\x00" in raw:
        raise DocumentError("The document is not text.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DocumentError("The document is not valid UTF-8 text.") from None


def _visible_lines(text: str) -> list[str]:
    """Remove comments, which a reader never sees."""
    normal_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    return _without_front_matter(_without_comments(normal_endings).split("\n"))


def _without_comments(text: str) -> str:
    kept: list[str] = []
    position = 0
    while (start := text.find("<!--", position)) != -1:
        end = text.find("-->", start + len("<!--"))
        if end == -1:
            break
        kept.append(text[position:start])
        position = end + len("-->")
    kept.append(text[position:])
    return "".join(kept)


def _without_front_matter(lines: list[str]) -> list[str]:
    if lines[0].rstrip() != "---":
        return lines
    for number, line in enumerate(lines[1:], start=1):
        if line.rstrip() in {"---", "..."}:
            return lines[number + 1 :]
    return lines


def _split_at_headings(lines: list[str]) -> list[tuple[str, list[str]]]:
    parts: list[tuple[str, list[str]]] = [(UNTITLED, [])]
    fence: str | None = None
    for line in lines:
        body = parts[-1][1]
        if fence is not None:
            if line.strip() and set(line.strip()) == {fence[0]} and len(line.strip()) >= len(fence):
                fence = None
            body.append(line)
        elif opening := _FENCE.match(line):
            fence = opening[1]
            body.append(line)
        elif marker := _HASH_HEADING.match(line):
            parts.append((_heading_text(line[marker.end() :]) or UNTITLED, []))
        elif _UNDERLINE.match(line) and _ends_with_one_line_paragraph(body):
            parts.append((body.pop().strip(), []))
        else:
            body.append(line)
    return parts


def _heading_text(after_marker: str) -> str:
    """Trim spaces and closing hashes, as in `## Title ##`, but not the `#` in `## C#`."""
    text = after_marker.strip(" \t")
    without_closing = text.rstrip("#")
    if not without_closing or without_closing[-1] in " \t":
        return without_closing.rstrip(" \t")
    return text


def _ends_with_one_line_paragraph(body: list[str]) -> bool:
    if not body or not body[-1].strip():
        return False
    return len(body) == 1 or not body[-2].strip()


def _to_sections(parts: list[tuple[str, list[str]]]) -> tuple[Section, ...]:
    sections: list[Section] = []
    used: set[str] = set()
    next_number: dict[str, int] = {}
    for heading, body in parts:
        text = "\n".join(body).strip()
        if not text:
            continue  # a heading with nothing under it, such as the document's title
        if len(sections) == MAX_SECTIONS:
            raise DocumentError(f"The document has more than {MAX_SECTIONS} sections.")
        if len(text) > MAX_SECTION_CHARS:
            raise DocumentError(
                f"Section {len(sections) + 1} is longer than {MAX_SECTION_CHARS} characters."
            )
        if len(heading) > MAX_HEADING_CHARS:
            raise DocumentError(
                f"The heading of section {len(sections) + 1} is longer than "
                f"{MAX_HEADING_CHARS} characters."
            )
        unique = _unique_heading(heading, used, next_number)
        used.add(unique)
        sections.append(Section(heading=unique, text=text))
    return tuple(sections)


def _unique_heading(heading: str, used: set[str], next_number: dict[str, int]) -> str:
    """Number a repeated heading. `next_number` holds the next number for each heading."""
    candidate = heading
    number = next_number.get(heading, 1)
    while candidate in used:
        number += 1
        candidate = f"{heading} ({number})"
    next_number[heading] = number
    return candidate

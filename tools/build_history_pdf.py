"""Builds docs/project-history.pdf from docs/project-history.md.

Run from the repo root: uv run --no-project --with reportlab python tools/build_history_pdf.py
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SOURCE = Path("docs/project-history.md")
TARGET = Path("docs/project-history.pdf")
PAGE_WIDTH = 174 * mm

INK = colors.HexColor("#1c1c1a")
LINE = colors.HexColor("#d8d8d2")
_BASE = getSampleStyleSheet()["BodyText"]
BODY = ParagraphStyle(
    "body", parent=_BASE, fontName="Helvetica", fontSize=10.3, leading=14.6, textColor=INK
)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=23, leading=27)
SECTION = ParagraphStyle(
    "section",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=14.5,
    leading=18,
    spaceBefore=13,
    spaceAfter=6,
    keepWithNext=1,
)
PART = ParagraphStyle(
    "part",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    spaceBefore=8,
    spaceAfter=3,
    keepWithNext=1,
)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=13, bulletIndent=2, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)


def inline(text: str) -> str:
    """Turn the Markdown this file uses into the small markup that PDF paragraphs understand."""
    marked = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))
    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", marked)


def build_table(rows: list[list[str]]) -> Table:
    """The first row is the header. Column widths follow how much text each column holds."""
    header, *body = rows
    longest = [max(len(row[n]) for row in rows) for n in range(len(header))]
    shares = [min(max(size, 8), 60) for size in longest]
    widths = [PAGE_WIDTH * share / sum(shares) for share in shares]
    content = [[Paragraph(inline(cell), HEAD) for cell in header]]
    content += [[Paragraph(inline(cell), CELL) for cell in row] for row in body]
    table = Table(content, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def table_rows(lines: list[str]) -> list[list[str]]:
    """Read Markdown table lines, leaving out the `|---|---|` line."""
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    return [row for row in rows if not all(set(cell) <= {"-", ":"} for cell in row)]


def to_flowables(markdown: str) -> list[Flowable]:
    story: list[Flowable] = []
    table_lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            story += [build_table(table_rows(table_lines)), Spacer(1, 7)]
            table_lines = []
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), PART))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), SECTION))
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), TITLE))
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), BULLET, bulletText="•"))
        elif line.strip():
            story.append(Paragraph(inline(line), BODY))
    if table_lines:
        story.append(build_table(table_rows(table_lines)))
    return story


def add_page_number(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    document = SimpleDocTemplate(
        str(TARGET),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="Project history",
    )
    story = to_flowables(SOURCE.read_text(encoding="utf-8"))
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    main()

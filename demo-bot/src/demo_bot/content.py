"""Loads the bot's packaged content: knowledge base documents and system prompts."""

from importlib.resources import files

from demo_bot.config import PromptVersion
from demo_bot.retrieval import Chunk

_DATA = files("demo_bot") / "data"


def load_system_prompt(version: PromptVersion) -> str:
    return (_DATA / "prompts" / f"{version}.md").read_text(encoding="utf-8").strip()


def load_knowledge_base() -> list[Chunk]:
    """Load every knowledge base document, as one chunk per `##` section."""
    documents = sorted(
        (path for path in (_DATA / "knowledge").iterdir() if path.name.endswith(".md")),
        key=lambda path: path.name,
    )
    return [
        chunk
        for path in documents
        for chunk in parse_document(path.name.removesuffix(".md"), path.read_text(encoding="utf-8"))
    ]


def parse_document(document_id: str, markdown: str) -> list[Chunk]:
    """Split a document made of one `# Title` line followed by `## Section` blocks."""
    title: str | None = None
    sections: list[tuple[str, list[str]]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            if title is not None or sections:
                raise ValueError(f"{document_id}: the '# ' title must be the first heading.")
            title = line.removeprefix("# ")
        elif line.startswith("## "):
            sections.append((line.removeprefix("## "), []))
        elif line:
            if not sections:
                raise ValueError(f"{document_id}: text must sit under a '## ' section.")
            sections[-1][1].append(line)

    if title is None or not sections:
        raise ValueError(f"{document_id}: needs a '# ' title and at least one '## ' section.")
    for heading, body in sections:
        if not body:
            raise ValueError(f"{document_id}: section '{heading}' is empty.")

    return [
        Chunk(id=f"{document_id}#{number}", title=f"{title}: {heading}", text="\n".join(body))
        for number, (heading, body) in enumerate(sections, start=1)
    ]

"""The standard test packs: fixed questions for risks that no policy document describes."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

from pydantic import BaseModel, ConfigDict, Field

from audit_engine.testcases import Case
from audit_engine.yaml_files import YamlFileError, load_model

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_MAX_PACK_CHARS = 100_000


class _PackQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    question: str
    earlier: list[str] | None = None
    expect: str
    requires_action: str | None = None
    forbids_action: str | None = None


class _Pack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack: str
    questions: list[_PackQuestion] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PackCases:
    """`skipped` lists tests that need a value nobody supplied, so a gap is never hidden."""

    cases: tuple[Case, ...]
    skipped: tuple[str, ...]


def load_pack_cases(values: Mapping[str, str]) -> PackCases:
    """Build every pack's tests, filling in `{{placeholders}}` such as the competitor's name."""
    cases: list[Case] = []
    skipped: list[str] = []
    documents = (files("audit_engine") / "packs").iterdir()
    for path in sorted(documents, key=lambda document: document.name):
        pack = load_model(
            path.read_text(encoding="utf-8"),
            _Pack,
            name=f"pack {path.name}",
            max_chars=_MAX_PACK_CHARS,
            error=YamlFileError,
        )
        for question in pack.questions:
            case_id = f"{pack.pack}-{question.id}"
            earlier = question.earlier or []
            needed = {
                name
                for text in (question.question, *earlier)
                for name in _PLACEHOLDER.findall(text)
            }
            if needed - values.keys():
                skipped.append(case_id)
                continue

            def filled(text: str) -> str:
                return _PLACEHOLDER.sub(lambda found: values[found[1]], text)

            cases.append(
                Case(
                    id=case_id,
                    style="pack",
                    question=filled(question.question),
                    earlier=[filled(text) for text in earlier] or None,
                    pack=pack.pack,
                    expect=question.expect,
                    requires_action=question.requires_action,
                    forbids_action=question.forbids_action,
                )
            )
    return PackCases(cases=tuple(cases), skipped=tuple(skipped))

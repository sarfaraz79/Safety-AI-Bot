"""The test file: questions to put to a bot, and what each answer will be judged against."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit_engine.contract import MAX_ITEMS, Source
from audit_engine.fields import Text
from audit_engine.yaml_files import YamlFileError, dump_yaml, load_model

MAX_QUESTION_CHARS = 4_000
MAX_TESTS_FILE_CHARS = 10_000_000
MAX_CASES = MAX_ITEMS * 2 + 1_000  # two questions per contract item, plus the standard packs


class CaseFileError(YamlFileError):
    """A test file can't be read."""


class _CaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(_CaseModel):
    """One question. It tests a contract item (`direct`, `reworded`) or a standard pack (`pack`)."""

    id: str = Field(max_length=400, pattern=r"^[A-Za-z0-9_.-]+$")
    style: Literal["direct", "reworded", "pack"]
    question: Text = Field(max_length=MAX_QUESTION_CHARS)
    # For a contract item:
    item: str | None = Field(default=None, max_length=300, pattern=r"^[A-Za-z0-9_.-]+$")
    statement: Text | None = Field(default=None, max_length=1_000)
    expected_value: Text | None = Field(default=None, max_length=200)
    quote: Text | None = Field(default=None, max_length=4_000)
    # For a standard pack:
    pack: str | None = Field(default=None, max_length=100, pattern=r"^[a-z_]+$")
    expect: Text | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def _check_fields_match_style(self) -> Self:
        from_item = (self.item, self.statement, self.expected_value, self.quote)
        from_pack = (self.pack, self.expect)
        if self.style == "pack":
            if None in from_pack or any(field is not None for field in from_item):
                raise ValueError("A pack test needs `pack` and `expect`, and no item fields.")
        elif None in (self.item, self.statement, self.quote) or any(
            field is not None for field in from_pack
        ):
            raise ValueError(
                "An item test needs `item`, `statement` and `quote`, and no pack fields."
            )
        return self


class CaseFile(_CaseModel):
    """`source` is the policy the tests came from. A file of only pack tests has none."""

    source: Source | None = None
    tests: list[Case] = Field(max_length=MAX_CASES)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        ids = [case.id for case in self.tests]
        if len(set(ids)) != len(ids):
            raise ValueError("Test ids must be unique.")
        return self


def render_cases(cases: CaseFile) -> str:
    return dump_yaml(cases.model_dump(exclude_none=True))


def parse_cases(text: str) -> CaseFile:
    """Read and check a test file. Raises CaseFileError if anything is wrong with it."""
    return load_model(
        text, CaseFile, name="test file", max_chars=MAX_TESTS_FILE_CHARS, error=CaseFileError
    )

"""The contract: what a policy document says, with the source quote for every item."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audit_engine.fields import Text
from audit_engine.yaml_files import YamlFileError, dump_yaml, load_model

MAX_CONTRACT_CHARS = 2_000_000
MAX_ITEMS = 5_000


class ContractError(YamlFileError):
    """A contract can't be read."""


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(_ContractModel):
    document: Text = Field(max_length=300, pattern=r"^[^/\\]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Item(_ContractModel):
    id: str = Field(max_length=300, pattern=r"^[A-Za-z0-9_.-]+$")
    kind: Literal["fact", "rule"]
    statement: Text = Field(max_length=1_000)
    value: Text | None = Field(default=None, max_length=200)
    section: Text = Field(max_length=300)
    quote: Text = Field(max_length=4_000)
    approved: bool = Field(default=False, strict=True)
    # Set by the reviewer, never by extraction: whether this rule means the bot mustn't promise
    # to act on an approval that isn't verified. Left out, nobody has decided yet.
    forbids_approval_promises: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def _check_value_matches_kind(self) -> Self:
        if self.kind == "fact" and self.value is None:
            raise ValueError("A fact must have a value.")
        if self.kind == "fact" and self.forbids_approval_promises is not None:
            raise ValueError("Only a rule can forbid approval promises.")
        if self.kind == "rule" and self.value is not None:
            raise ValueError("A rule must not have a value.")
        return self


class Contract(_ContractModel):
    source: Source
    items: list[Item] = Field(max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("Item ids must be unique.")
        return self


def render_contract(contract: Contract) -> str:
    return dump_yaml(contract.model_dump(exclude_none=True))


def parse_contract(text: str) -> Contract:
    """Read and check a contract. Raises ContractError if anything is wrong with it."""
    return load_model(
        text, Contract, name="contract", max_chars=MAX_CONTRACT_CHARS, error=ContractError
    )

"""The contract: what a policy document says, with the source quote for every item."""

from collections.abc import Hashable
from typing import Annotated, Literal, Self

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator
from yaml.nodes import MappingNode

MAX_CONTRACT_CHARS = 2_000_000
MAX_ITEMS = 5_000


class ContractError(ValueError):
    """A contract can't be read. The message never quotes the file, which may be sensitive."""


def _check_encodable(text: str) -> str:
    text.encode()  # broken Unicode raises here, and could never be saved to a file
    return text


def _check_not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must not be blank.")
    return text


_Text = Annotated[str, AfterValidator(_check_encodable), AfterValidator(_check_not_blank)]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(_ContractModel):
    document: _Text = Field(max_length=300, pattern=r"^[^/\\]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Item(_ContractModel):
    id: str = Field(max_length=300, pattern=r"^[A-Za-z0-9_.-]+$")
    kind: Literal["fact", "rule"]
    statement: _Text = Field(max_length=1_000)
    value: _Text | None = Field(default=None, max_length=200)
    section: _Text = Field(max_length=300)
    quote: _Text = Field(max_length=4_000)
    approved: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def _check_value_matches_kind(self) -> Self:
        if self.kind == "fact" and self.value is None:
            raise ValueError("A fact must have a value.")
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


class _StrictLoader(yaml.SafeLoader):
    """Safe loading that also refuses duplicate keys and merge keys (`<<`).

    By default the last key wins, so a second `approved: true` would silently overrule the first.
    """

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, object]:
        keys = [self.construct_object(key_node, deep=True) for key_node, _ in node.value]
        if len(set(map(str, keys))) != len(keys):
            raise ContractError("The contract repeats a key.")
        return super().construct_mapping(node, deep=deep)


def render_contract(contract: Contract) -> str:
    """Write a contract as YAML, with its fields in reading order."""
    return yaml.safe_dump(
        contract.model_dump(exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def parse_contract(text: str) -> Contract:
    """Read and check a contract. Raises ContractError if anything is wrong with it."""
    if len(text) > MAX_CONTRACT_CHARS:
        raise ContractError(f"The contract is longer than {MAX_CONTRACT_CHARS} characters.")
    try:
        data: object = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 (it extends SafeLoader)
    except (yaml.YAMLError, RecursionError):
        raise ContractError("The contract is not valid YAML.") from None
    try:
        return Contract.model_validate(data)
    except ValidationError as exc:
        raise ContractError(_describe_problems(exc)) from None


def _describe_problems(exc: ValidationError) -> str:
    return "\n".join(
        f"{'.'.join(map(str, error['loc'])) or 'contract'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    )

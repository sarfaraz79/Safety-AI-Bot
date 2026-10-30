"""Reads and writes the engine's YAML files, which people review and edit by hand."""

from collections.abc import Hashable

import yaml
from pydantic import BaseModel, ValidationError
from yaml.nodes import MappingNode


class YamlFileError(ValueError):
    """A file can't be read. The message never quotes the file, which may be sensitive."""


class _RepeatedKeyError(Exception):
    pass


class _StrictLoader(yaml.SafeLoader):
    """Safe loading that also refuses duplicate keys and merge keys (`<<`).

    By default the last key wins, so a second `approved: true` would silently overrule the first.
    """

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, object]:
        keys = [self.construct_object(key_node, deep=True) for key_node, _ in node.value]
        if len(set(map(str, keys))) != len(keys):
            raise _RepeatedKeyError
        return super().construct_mapping(node, deep=deep)


def dump_yaml(data: object) -> str:
    """Write data as YAML, keeping fields in reading order and text in its own language."""
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)


def load_model[ModelT: BaseModel](
    text: str, model: type[ModelT], *, name: str, max_chars: int, error: type[YamlFileError]
) -> ModelT:
    """Read and check a file. `name` is how messages refer to it, as in "The contract is..."."""
    if len(text) > max_chars:
        raise error(f"The {name} is longer than {max_chars} characters.")
    try:
        data: object = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 (it extends SafeLoader)
    except _RepeatedKeyError:
        raise error(f"The {name} repeats a key.") from None
    except (yaml.YAMLError, RecursionError):
        raise error(f"The {name} is not valid YAML.") from None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise error(_describe_problems(exc, name)) from None


def _describe_problems(exc: ValidationError, name: str) -> str:
    return "\n".join(
        f"{'.'.join(map(str, problem['loc'])) or name}: {problem['msg']}"
        for problem in exc.errors(include_url=False, include_input=False)
    )

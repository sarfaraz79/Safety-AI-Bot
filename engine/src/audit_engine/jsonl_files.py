"""Reads the engine's line-by-line JSON files, which are written as results arrive."""

from pydantic import BaseModel, ValidationError

from audit_engine.yaml_files import YamlFileError


def parse_jsonl[ModelT: BaseModel](
    text: str, model: type[ModelT], *, name: str, max_chars: int, error: type[YamlFileError]
) -> tuple[list[ModelT], bool]:
    """Return the entries, and whether a crash cut the file short.

    Only a last line with no line break can be a cut-off write. Anything else that's wrong
    means the file has been damaged or edited, so it's refused.
    """
    if len(text) > max_chars:
        raise error(f"The {name} is longer than {max_chars} characters.")
    lines = text.split("\n")
    last = lines.pop()
    entries: list[ModelT] = []
    for number, line in enumerate(lines, start=1):
        try:
            entries.append(model.model_validate_json(line))
        except ValidationError:
            raise error(f"Line {number} of the {name} is not valid.") from None
    if not last:
        return entries, False
    try:
        entries.append(model.model_validate_json(last))
    except ValidationError:
        return entries, True
    return entries, False

"""Field types shared by the engine's file formats."""

from typing import Annotated

from pydantic import AfterValidator


def _check_encodable(text: str) -> str:
    text.encode()  # broken Unicode raises here, and could never be saved to a file
    return text


def _check_not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must not be blank.")
    return text


Text = Annotated[str, AfterValidator(_check_encodable), AfterValidator(_check_not_blank)]

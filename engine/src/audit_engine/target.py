"""The target file, and talking to the bot it describes. Any bot's reply is untrusted data."""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import httpx2
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from audit_engine.config import is_this_computer
from audit_engine.yaml_files import YamlFileError, load_model

MAX_TARGET_CHARS = 100_000
MAX_RESPONSE_BYTES = 1_000_000
QUESTION_PLACEHOLDER = "{{question}}"

_PATH = r"^[^.\s]+(\.[^.\s]+)*$"
_HEADER_SAFE_TOKEN = re.compile(r"[!-~]+")
_SUCCESS = range(200, 300)


class TargetError(YamlFileError):
    """A target can't be used. The message never shows a token's value."""


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    request: dict[str, Any]
    reply_path: str = Field(pattern=_PATH)
    capture: dict[str, str] = Field(default_factory=dict)
    token_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]*$")
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    @field_validator("url")
    @classmethod
    def _check_url_is_encrypted(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https" and not is_this_computer((value.host or "").strip("[]")):
            raise ValueError("must use https unless it points at this computer.")
        return value

    @field_validator("capture")
    @classmethod
    def _check_capture_paths(cls, value: dict[str, str]) -> dict[str, str]:
        if not all(re.fullmatch(_PATH, path) for path in value.values()):
            raise ValueError("Each capture must be a path such as `data.tool_calls`.")
        return value

    @model_validator(mode="after")
    def _check_question_has_a_place(self) -> Self:
        if _fill(self.request, "") == self.request:
            raise ValueError(f"`request` must contain {QUESTION_PLACEHOLDER}.")
        return self


@dataclass(frozen=True, slots=True)
class Answer:
    """What one question got. `error` is set instead of `reply` when no usable reply came back."""

    reply: str | None
    captured: dict[str, Any]
    status: int | None
    error: str | None


def parse_target(text: str) -> Target:
    """Read and check a target file. Raises TargetError if anything is wrong with it."""
    return load_model(text, Target, name="target", max_chars=MAX_TARGET_CHARS, error=TargetError)


def read_token(target: Target, environment: Mapping[str, str]) -> str | None:
    """Fetch the token named by the target. The file only ever holds the variable's name."""
    if target.token_env is None:
        return None
    token = environment.get(target.token_env)
    if not token:
        raise TargetError(f"The environment variable {target.token_env} is not set.")
    if not _HEADER_SAFE_TOKEN.fullmatch(token):
        raise TargetError(
            f"{target.token_env} must contain only printable ASCII characters, with no spaces."
        )
    return token


class TargetClient:
    def __init__(
        self, target: Target, token: str | None, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        self._target = target
        self._url = str(target.url)
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        # Redirects stay off, so a bot can't send the token somewhere else.
        self._client = http_client or httpx2.AsyncClient(follow_redirects=False)

    async def ask(self, question: str) -> Answer:
        """Put one question to the bot. Every failure is returned as an Answer, never raised."""
        try:
            async with asyncio.timeout(self._target.timeout_seconds):
                status, body = await self._post(question)
        except (TimeoutError, httpx2.TimeoutException):
            return _failed(None, "The request timed out.")
        except httpx2.HTTPError:
            return _failed(None, "The request failed.")
        if body is None:
            return _failed(status, "The response was too large.")
        if status not in _SUCCESS:
            return _failed(status, f"The bot returned HTTP {status}.")
        return self._read_answer(status, body)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, question: str) -> tuple[int, bytes | None]:
        """Read the response up to the limit, in pieces, so a huge one can't exhaust memory."""
        payload = _fill(self._target.request, question)
        async with self._client.stream(
            "POST", self._url, json=payload, headers=self._headers
        ) as response:
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    return response.status_code, None
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)

    def _read_answer(self, status: int, body: bytes) -> Answer:
        try:
            data = json.loads(body, parse_constant=_refuse_constant)
        except (ValueError, RecursionError):
            return _failed(status, "The response was not valid JSON.")
        reply = _read_path(data, self._target.reply_path)
        if not isinstance(reply, str):
            return _failed(status, f"The response has no text at {self._target.reply_path}.")
        captured = {name: _read_path(data, path) for name, path in self._target.capture.items()}
        try:
            json.dumps([reply, captured], ensure_ascii=False).encode()
        except ValueError:
            return _failed(status, "The response contains broken Unicode.")
        return Answer(reply=reply, captured=captured, status=status, error=None)


def _failed(status: int | None, error: str) -> Answer:
    return Answer(reply=None, captured={}, status=status, error=error)


def _refuse_constant(name: str) -> None:
    raise ValueError(name)  # NaN and Infinity aren't JSON, though Python would accept them


def _fill(template: object, question: str) -> Any:
    """Put the question in place as data, so its wording can't change the request's shape."""
    if isinstance(template, str):
        return template.replace(QUESTION_PLACEHOLDER, question)
    if isinstance(template, dict):
        return {key: _fill(value, question) for key, value in template.items()}
    if isinstance(template, list):
        return [_fill(value, question) for value in template]
    return template


def _read_path(data: object, path: str) -> object:
    """Follow a path such as `choices.0.message.content`. Returns None if it leads nowhere."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdecimal() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current

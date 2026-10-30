"""The boundary with the language model. Switching provider or SDK means changing only this file."""

from typing import Annotated, Protocol

import httpx2
import openai
from pydantic import AfterValidator, BaseModel, Field


class LLMError(Exception):
    """The model provider failed to return a usable response."""


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str:
        """Send one question and return the model's text, empty if it gave none."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def _check_encodable(text: str) -> str:
    text.encode()  # broken Unicode raises here, and could never be saved to a file
    return text


class _ProviderMessage(BaseModel):
    content: Annotated[str, AfterValidator(_check_encodable)] | None = None


class _ProviderChoice(BaseModel):
    message: _ProviderMessage


class _ProviderCompletion(BaseModel):
    """The parts of a provider's reply the engine relies on. The SDK itself doesn't enforce them."""

    choices: list[_ProviderChoice] = Field(min_length=1)


class OpenAICompatibleClient:
    """Talks to any provider that implements the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=2,
            http_client=http_client,
        )

    async def complete(self, system: str, user: str) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            reply = _ProviderCompletion.model_validate(completion, from_attributes=True)
        except (openai.OpenAIError, httpx2.HTTPError, ValueError) as exc:
            # Only the error's type: its text can quote the reply or the request.
            raise LLMError(type(exc).__name__) from exc
        return reply.choices[0].message.content or ""

    async def aclose(self) -> None:
        await self._client.close()

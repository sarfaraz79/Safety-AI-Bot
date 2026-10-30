"""Test doubles for running the engine without calling a real model."""

from audit_engine.llm import LLMError


class ScriptedLLM:
    """Plays back the given replies in order, and records every question it receives.

    An LLMError in the script is raised instead of returned, to simulate a provider failure.
    """

    def __init__(self, *script: str | LLMError) -> None:
        self._script = list(script)
        self.requests: list[tuple[str, str]] = []
        self.closed = False

    async def complete(self, system: str, user: str) -> str:
        self.requests.append((system, user))
        if not self._script:
            raise RuntimeError("ScriptedLLM has no replies left.")
        step = self._script.pop(0)
        if isinstance(step, LLMError):
            raise step
        return step

    async def aclose(self) -> None:
        self.closed = True

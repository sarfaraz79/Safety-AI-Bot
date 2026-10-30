import pytest

from audit_engine.llm import LLMError
from audit_engine.testing import ScriptedLLM

pytestmark = pytest.mark.anyio


async def test_plays_back_replies_and_records_questions() -> None:
    llm = ScriptedLLM("first", "second")

    assert await llm.complete("s1", "u1") == "first"
    assert await llm.complete("s2", "u2") == "second"
    assert llm.requests == [("s1", "u1"), ("s2", "u2")]


async def test_raises_scripted_failures() -> None:
    with pytest.raises(LLMError, match="RateLimitError"):
        await ScriptedLLM(LLMError("RateLimitError")).complete("s", "u")


async def test_running_out_of_replies_is_a_loud_failure() -> None:
    with pytest.raises(RuntimeError, match="no replies left"):
        await ScriptedLLM().complete("s", "u")


async def test_records_being_closed() -> None:
    llm = ScriptedLLM()

    await llm.aclose()

    assert llm.closed

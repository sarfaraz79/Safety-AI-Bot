import pytest

from demo_bot.llm import UserMessage
from demo_bot.testing import ScriptedLLM, reply

pytestmark = pytest.mark.anyio


async def test_scripted_llm_plays_back_in_order_then_fails_loudly() -> None:
    llm = ScriptedLLM(reply("first"), reply("second"))

    assert (await llm.complete([UserMessage("a")], [])).content == "first"
    assert (await llm.complete([UserMessage("b")], [])).content == "second"
    with pytest.raises(RuntimeError, match="no responses left"):
        await llm.complete([UserMessage("c")], [])
    assert [messages for messages, _ in llm.requests] == [
        [UserMessage("a")],
        [UserMessage("b")],
        [UserMessage("c")],
    ]

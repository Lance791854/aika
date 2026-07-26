"""Wake-gate behavior for the AIKA production agent.

Pure logic tests — on_user_turn_completed raises StopResponse to skip a turn,
so we can exercise the gate directly without STT/LLM. Two modes: "off" replies
to everything, "strict" only replies when the chef addresses AIKA by name.
"""

import sys
from pathlib import Path

import pytest
from livekit.agents import llm
from livekit.agents.llm import StopResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_aika


def _msg(text: str) -> llm.ChatMessage:
    return llm.ChatMessage(role="user", content=[text])


async def _replies(agent: agent_aika.Assistant, text: str) -> bool:
    """True if the agent would reply to `text`, False if the gate silences it."""
    try:
        await agent.on_user_turn_completed(None, _msg(text))
        return True
    except StopResponse:
        return False


@pytest.mark.asyncio
async def test_off_always_replies() -> None:
    a = agent_aika.Assistant(wake_mode="off")
    assert await _replies(a, "pass me the salt")
    assert await _replies(a, "how long for steak")


@pytest.mark.asyncio
async def test_strict_requires_wake_word_each_turn() -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "AIKA set a steak timer four minutes")
    assert not await _replies(a, "how long for the steak")
    assert await _replies(a, "AIKA how long for the steak")


@pytest.mark.asyncio
async def test_strict_accepts_hey_aika() -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "hey AIKA what was the freezer at")


@pytest.mark.asyncio
async def test_strict_silent_without_wake_word() -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    assert not await _replies(a, "pass me the salt")


@pytest.mark.asyncio
async def test_bare_wake_opens_window_for_next_turn() -> None:
    """Chefs naturally say "AIKA" then pause — the command lands in a second
    turn with no wake word. A bare address must open the gate for it."""
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "AIKA.")
    assert await _replies(a, "how long for the steak")


@pytest.mark.asyncio
async def test_hey_aika_alone_also_opens_window() -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "hey AIKA")
    assert await _replies(a, "set a steak timer four minutes")


@pytest.mark.asyncio
async def test_window_is_one_turn_only() -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "AIKA.")
    assert await _replies(a, "set a steak timer four minutes")
    assert not await _replies(a, "pass the salt")


@pytest.mark.asyncio
async def test_window_expires(monkeypatch) -> None:
    a = agent_aika.Assistant(wake_mode="strict")
    t = {"now": 1000.0}
    monkeypatch.setattr(agent_aika.time, "time", lambda: t["now"])
    assert await _replies(a, "AIKA.")
    t["now"] += agent_aika.Assistant.WAKE_WINDOW_S + 1
    assert not await _replies(a, "pass the salt")


@pytest.mark.asyncio
async def test_command_with_wake_word_does_not_open_window() -> None:
    """Only a bare address opens the window — a full command already got its
    reply, and side chatter right after must stay gated."""
    a = agent_aika.Assistant(wake_mode="strict")
    assert await _replies(a, "AIKA set a steak timer four minutes")
    assert not await _replies(a, "pass the salt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "the chicken is basically done",
        "what was the medical fridge at",
        "paprika and salt on the lamb",
        "can I get a replica of that order",
        "silica gel packet in the dry store",
    ],
)
async def test_strict_ignores_substring_lookalikes(phrase) -> None:
    """Wake words match whole words only — fragments like "ica"/"ika" buried
    in ordinary speech must not trip the gate open."""
    a = agent_aika.Assistant(wake_mode="strict")
    assert not await _replies(a, phrase)

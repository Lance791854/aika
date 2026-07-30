"""Wearable-sim wake gate ("device" mode).

Simulates the ESP32 duty cycle: a tiny self-hosted spotter hears everything,
and audio only reaches the real STT/LLM after the wake word. device_gate is
the pure forward/drop decision; the Assistant in device mode only has to drop
the empty turns the gate leaves behind.
"""

import sys
from pathlib import Path

import pytest
from livekit.agents import llm
from livekit.agents.llm import StopResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_aika

WINDOW = agent_aika.WAKE_SPOT_WINDOW_S


def test_dormant_utterance_is_dropped() -> None:
    forward, armed = agent_aika.device_gate("pass the salt", 0.0, now=1000.0)
    assert not forward
    assert armed == 0.0


def test_wake_word_forwards_and_arms() -> None:
    forward, armed = agent_aika.device_gate("aika steak four minutes", 0.0, now=1000.0)
    assert forward
    assert armed == 1000.0 + WINDOW


def test_sloppy_spotter_transcript_still_wakes() -> None:
    # tiny models mangle "AIKA" — the existing variant list must keep working
    forward, _ = agent_aika.device_gate("hey ayka", 0.0, now=1000.0)
    assert forward


def test_spelled_out_name_wakes() -> None:
    # tiny model sometimes writes the name as initials
    forward, _ = agent_aika.device_gate("A.I.K.A. Stake four minutes.", 0.0, now=1000.0)
    assert forward


def test_paprika_does_not_wake() -> None:
    forward, _ = agent_aika.device_gate("paprika on the lamb", 0.0, now=1000.0)
    assert not forward


def test_armed_window_forwards_without_wake_word() -> None:
    forward, armed = agent_aika.device_gate("four minutes for the lamb", 1005.0, now=1000.0)
    assert forward
    assert armed == 1000.0 + WINDOW  # follow-up re-arms the window


def test_expired_window_drops_again() -> None:
    forward, _ = agent_aika.device_gate("four minutes for the lamb", 1005.0, now=1006.0)
    assert not forward


@pytest.mark.asyncio
async def test_device_mode_drops_empty_turns() -> None:
    a = agent_aika.Assistant(wake_mode="device")
    with pytest.raises(StopResponse):
        await a.on_user_turn_completed(
            None, llm.ChatMessage(role="user", content=[""])
        )


@pytest.mark.asyncio
async def test_device_mode_passes_forwarded_turns() -> None:
    # gating already happened pre-STT; any real transcript goes through
    a = agent_aika.Assistant(wake_mode="device")
    await a.on_user_turn_completed(
        None, llm.ChatMessage(role="user", content=["steak four minutes"])
    )

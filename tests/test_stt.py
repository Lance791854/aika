"""STT engine selection in build_stt (agent_aika)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from livekit.plugins import cartesia, deepgram, openai

import agent_aika


def test_build_stt_cartesia_returns_cartesia_engine() -> None:
    stt = agent_aika.build_stt("cartesia")
    assert isinstance(stt, cartesia.STT)


def test_build_stt_cloud_is_deepgram_and_unknown_falls_back() -> None:
    assert isinstance(agent_aika.build_stt("cloud"), deepgram.STT)
    assert isinstance(agent_aika.build_stt("something-unknown"), deepgram.STT)


def test_build_stt_self_hosted_use_openai_wrapper() -> None:
    assert isinstance(agent_aika.build_stt("gpu"), openai.STT)
    assert isinstance(agent_aika.build_stt("local"), openai.STT)

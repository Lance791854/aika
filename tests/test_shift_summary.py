"""End-of-shift summary — the HACCP-style spoken report.

Two layers, same as unhandled: unit tests on storage.shift_summary (what goes
into the report, out-of-range readings flagged) and a behavioral test that the
LLM reaches for the tool when a chef asks for the summary.
"""

import os
import sys
from pathlib import Path

import pytest
from livekit.agents import AgentSession
from livekit.plugins import groq, openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_aika
import storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Point storage at a throwaway file so tests never touch data/aika.json."""
    monkeypatch.setattr(storage, "DATA_PATH", tmp_path / "aika.json")


def _llm():
    """Judge LLM. Groq free tier by default; set AIKA_TEST_LLM=cf to use
    Cloudflare when Groq's daily cap runs out."""
    if os.environ.get("AIKA_TEST_LLM") == "cf":
        return openai.LLM(
            base_url=agent_aika.CF_BASE_URL,
            api_key=agent_aika.CF_API_TOKEN,
            model=agent_aika.CF_LLM_MODEL,
        )
    return groq.LLM(model="llama-3.3-70b-versatile")


# --- storage layer (fast, deterministic) ---


def test_empty_shift_says_so() -> None:
    text = storage.shift_summary()
    assert "No temperatures logged" in text
    assert "No notes" in text


def test_summary_covers_temps_and_notes() -> None:
    storage.add_temperature("freezer", -20)
    storage.add_temperature("fridge", 4)
    storage.add_note("out of butter")
    text = storage.shift_summary()
    assert "freezer at -20" in text
    assert "fridge at 4" in text
    assert "out of butter" in text
    assert "All in range" in text


def test_summary_flags_out_of_range_readings() -> None:
    storage.add_temperature("freezer", -12)
    text = storage.shift_summary()
    assert "Warning" in text
    assert "above safe range" in text


def test_summary_uses_latest_reading_per_location() -> None:
    storage.add_temperature("freezer", -12)
    storage.add_temperature("freezer", -20)
    text = storage.shift_summary()
    assert "freezer at -20" in text
    assert "Warning" not in text


def test_summary_mentions_unhandled_requests() -> None:
    storage.add_unhandled("order more beef")
    storage.add_unhandled("play some music")
    assert "2 requests" in storage.shift_summary()


# --- behavioral (LLM chooses the tool) ---


@pytest.mark.asyncio
async def test_asking_for_summary_calls_the_tool() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(user_input="AIKA give me the shift summary")
        result.expect.contains_function_call(name="shift_summary")

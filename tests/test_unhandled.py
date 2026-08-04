"""Unhandled-request logging for the AIKA production agent.

Two layers:
- storage unit tests (deterministic, no LLM) for add/recent_unhandled
- behavioral tests where the LLM must decide to call log_unhandled for a
  request AIKA can't fulfil, and must NOT call it for a normal one.
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


def test_add_and_recent_unhandled_newest_first() -> None:
    assert storage.recent_unhandled() == []
    storage.add_unhandled("order more beef")
    storage.add_unhandled("turn on the oven")
    recent = storage.recent_unhandled()
    assert [e["text"] for e in recent] == ["turn on the oven", "order more beef"]
    assert all("at" in e for e in recent)


def test_recent_unhandled_respects_limit() -> None:
    for i in range(8):
        storage.add_unhandled(f"request {i}")
    assert len(storage.recent_unhandled(5)) == 5


def test_clear_unhandled_empties_the_log() -> None:
    storage.add_unhandled("order more beef")
    storage.add_unhandled("turn on the oven")
    assert storage.recent_unhandled()
    storage.clear_unhandled()
    assert storage.recent_unhandled() == []


def test_delete_unhandled_removes_only_the_matching_entry() -> None:
    storage.add_unhandled("keep me")
    storage.add_unhandled("delete me")
    target = next(e for e in storage.recent_unhandled() if e["text"] == "delete me")
    storage.delete_unhandled(target["at"])
    assert [e["text"] for e in storage.recent_unhandled()] == ["keep me"]


def test_delete_note_and_clear_notes() -> None:
    storage.add_note("out of butter")
    storage.add_note("defrost lamb")
    target = next(n for n in storage.recent_notes() if n["text"] == "out of butter")
    storage.delete_note(target["at"])
    assert [n["text"] for n in storage.recent_notes()] == ["defrost lamb"]
    storage.clear_notes()
    assert storage.recent_notes() == []


def test_delete_temperature_location_and_clear() -> None:
    storage.add_temperature("fridge", 4)
    storage.add_temperature("freezer", -20)
    storage.add_temperature("fridge", 6)  # second fridge reading
    storage.delete_temperature_location("fridge")
    locs = [t["location"] for t in storage.recent_temperatures()]
    assert "fridge" not in locs and "freezer" in locs
    storage.clear_temperatures()
    assert storage.recent_temperatures() == []


# --- behavioral (LLM chooses the tool) ---


@pytest.mark.asyncio
async def test_logs_unsupported_request() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(
            user_input="AIKA, order another case of beef from the supplier"
        )
        result.expect.contains_function_call(name="log_unhandled")
    assert storage.recent_unhandled(), "unsupported request should be logged"


@pytest.mark.asyncio
async def test_normal_request_not_logged_as_unhandled() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(user_input="make a note we're out of butter")
        result.expect.contains_function_call(name="add_note")
    assert storage.recent_unhandled() == [], (
        "a normal note request must not be logged as unhandled"
    )

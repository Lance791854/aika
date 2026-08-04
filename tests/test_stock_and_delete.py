"""Stock requests + voice deletion with confirmation.

Storage unit tests are fast and deterministic. The behavioral tests pin the
two risky LLM decisions: routing "we're low on X" to request_stock (not
add_note), and never deleting until the chef says yes.
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


def test_add_and_recent_stock_newest_first() -> None:
    storage.add_stock("cream", "five kilos", "urgent")
    storage.add_stock("butter")
    recent = storage.recent_stock()
    assert [e["item"] for e in recent] == ["butter", "cream"]
    assert recent[1]["quantity"] == "five kilos"
    assert recent[1]["urgency"] == "urgent"
    assert recent[0]["urgency"] == "normal"


def test_delete_and_clear_stock() -> None:
    storage.add_stock("cream")
    storage.add_stock("butter")
    target = next(e for e in storage.recent_stock() if e["item"] == "cream")
    storage.delete_stock(target["at"])
    assert [e["item"] for e in storage.recent_stock()] == ["butter"]
    storage.clear_stock()
    assert storage.recent_stock() == []


def test_shift_summary_includes_stock() -> None:
    storage.add_stock("cream", "five kilos", "urgent")
    text = storage.shift_summary()
    assert "cream" in text
    assert "urgent" in text


def test_scope_name_cleans_input() -> None:
    assert storage.scope_name("Lance") == "lance"
    assert storage.scope_name("Chef Marco!") == "chef-marco"
    assert storage.scope_name("  ") == ""
    assert len(storage.scope_name("x" * 99)) == 24


def test_set_scope_isolates_chefs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage, "_BASE_PATH", tmp_path / "aika.json")
    storage.set_scope("lance")
    storage.add_note("lance's note")
    storage.set_scope("marco")
    assert storage.recent_notes() == []
    storage.add_note("marco's note")
    storage.set_scope("lance")
    assert [n["text"] for n in storage.recent_notes()] == ["lance's note"]
    storage.set_scope("")  # back to the shared default
    assert storage.recent_notes() == []


# --- behavioral (LLM chooses the tool) ---


@pytest.mark.asyncio
async def test_low_on_something_becomes_stock_request() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(
            user_input="AIKA we're low on cream, order five kilos, it's urgent"
        )
        result.expect.contains_function_call(name="request_stock")
    items = [e["item"].lower() for e in storage.recent_stock()]
    assert any("cream" in i for i in items)


@pytest.mark.asyncio
async def test_explicit_note_stays_a_note() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(user_input="make a note we're out of butter")
        result.expect.contains_function_call(name="add_note")
    assert storage.recent_stock() == []


@pytest.mark.asyncio
async def test_asking_whats_needed_lists_stock() -> None:
    storage.add_stock("cream", "five kilos", "urgent")
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(user_input="AIKA what stock do we need")
        result.expect.contains_function_call(name="list_stock")


@pytest.mark.asyncio
async def test_delete_note_waits_for_yes() -> None:
    storage.add_note("out of butter")
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(user_input="AIKA delete the note about butter")
        result.expect.contains_function_call(name="delete_note")
        # nothing deleted yet — confirmation pending
        assert [n["text"] for n in storage.recent_notes()] == ["out of butter"]
        result = await session.run(user_input="yes")
        result.expect.contains_function_call(name="confirm_delete")
    assert storage.recent_notes() == []


@pytest.mark.asyncio
async def test_delete_note_no_keeps_it() -> None:
    storage.add_note("out of butter")
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        await session.run(user_input="AIKA delete the note about butter")
        result = await session.run(user_input="no, keep it")
        result.expect.contains_function_call(name="confirm_delete")
    assert [n["text"] for n in storage.recent_notes()] == ["out of butter"]

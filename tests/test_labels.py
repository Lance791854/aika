"""Food safety labels — proof of concept.

Unit tests cover the use-by date math, one behavioral test covers the LLM
picking the tool.
"""

import os
import sys
import time
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


def test_label_use_by_defaults_to_three_days() -> None:
    storage.add_label("pesto")
    (label,) = storage.recent_labels()
    assert label["item"] == "pesto"
    assert abs(label["use_by"] - (time.time() + 3 * 86400)) < 60


def test_label_use_by_follows_given_days() -> None:
    storage.add_label("cooked chicken", days=1)
    (label,) = storage.recent_labels()
    assert abs(label["use_by"] - (time.time() + 86400)) < 60


def test_delete_and_clear_labels() -> None:
    storage.add_label("pesto")
    storage.add_label("aioli")
    target = next(e for e in storage.recent_labels() if e["item"] == "pesto")
    storage.delete_label(target["at"])
    assert [e["item"] for e in storage.recent_labels()] == ["aioli"]
    storage.clear_labels()
    assert storage.recent_labels() == []


# --- behavioral (LLM chooses the tool) ---


@pytest.mark.asyncio
async def test_label_without_time_asks_first() -> None:
    # no days given — AIKA should ask, not make the label yet
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        await session.run(user_input="AIKA make a label for the pesto")
    assert storage.recent_labels() == []


@pytest.mark.asyncio
async def test_asking_for_a_label_calls_the_tool() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        await session.start(agent_aika.Assistant())
        result = await session.run(
            user_input="AIKA make a label for the pesto, three days"
        )
        result.expect.contains_function_call(name="make_label")
    items = [e["item"].lower() for e in storage.recent_labels()]
    assert any("pesto" in i for i in items)

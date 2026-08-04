"""Timed reminders for the AIKA production agent.

Routing is the risky part: "remind me in 20 minutes to X" must become a
set_reminder (spoken aloud when time is up), while "remind the morning shift
to X" has no time and must stay a note. Behavioral tests pin that down.
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


@pytest.mark.asyncio
async def test_timed_reminder_uses_set_reminder() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        agent = agent_aika.Assistant()
        await session.start(agent)
        result = await session.run(
            user_input="AIKA remind me in 20 minutes to rotate the stock"
        )
        result.expect.contains_function_call(name="set_reminder")
    assert len(agent.reminders) == 1
    (reminder,) = agent.reminders.values()
    assert abs(reminder["end_time"] - (time.time() + 20 * 60)) < 60


@pytest.mark.asyncio
async def test_reminder_without_a_task_still_sets() -> None:
    # "set reminder two minutes" — no task given. Set it anyway, don't ask.
    async with _llm() as llm, AgentSession(llm=llm) as session:
        agent = agent_aika.Assistant()
        await session.start(agent)
        result = await session.run(user_input="AIKA set a reminder for two minutes")
        result.expect.contains_function_call(name="set_reminder")
    assert len(agent.reminders) == 1


@pytest.mark.asyncio
async def test_untimed_reminder_stays_a_note() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        agent = agent_aika.Assistant()
        await session.start(agent)
        result = await session.run(
            user_input="remind the morning shift to defrost the lamb"
        )
        result.expect.contains_function_call(name="add_note")
    assert agent.reminders == {}
    assert storage.recent_notes(), "untimed reminder should be saved as a note"


@pytest.mark.asyncio
async def test_check_reminders_reports_whats_set() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        agent = agent_aika.Assistant()
        await session.start(agent)
        await session.run(user_input="AIKA remind me in 10 minutes to stir the sauce")
        result = await session.run(user_input="AIKA what reminders are set")
        result.expect.contains_function_call(name="check_reminders")


@pytest.mark.asyncio
async def test_cancel_reminder_by_keyword() -> None:
    async with _llm() as llm, AgentSession(llm=llm) as session:
        agent = agent_aika.Assistant()
        await session.start(agent)
        await session.run(user_input="AIKA remind me in 10 minutes to stir the sauce")
        assert len(agent.reminders) == 1
        result = await session.run(user_input="AIKA cancel the sauce reminder")
        result.expect.contains_function_call(name="cancel_reminder")
    assert agent.reminders == {}

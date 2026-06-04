"""AIKA production agent — per-call switching between cloud and self-hosted
inference via RoomConfiguration.agents[0].metadata.

Frontend sends a JSON metadata blob like {"stt":"cloud","llm":"local","tts":"cloud"};
this agent reads it on join and builds the appropriate plugin instances.

This is the deployed worker. See src/agent.py for the clean cloud-only
reference template, and src/agent_local.py for the local-stack debug variant
with side-by-side STT comparison.
"""

import asyncio
import json
import logging
import time

import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    tts as livekit_tts,
)
from livekit.agents.llm import StopResponse
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, groq, noise_cancellation, openai, silero

logger = logging.getLogger("agent_aika")

load_dotenv(".env.local")


# ---------------------------------------------------------------------------
# Self-hosted inference endpoints. The DOCKER-USER iptables rule on
# <inference-host> allows traffic to these ports ONLY from this VPS
# (<frontend-host>). Anyone else gets dropped.
# ---------------------------------------------------------------------------
LOCAL_OLLAMA_URL = "http://<inference-host>:11434/v1"
LOCAL_SPEACHES_URL = "http://<inference-host>:8000/v1"
LOCAL_LLM_MODEL = "aika-llm"  # custom variant of qwen2.5:3b with num_thread=8
# Multilingual whisper-medium is the realistic CPU floor for "works in demo":
# ~5-7s/utterance, accent-tolerant. large-v3 is 17-20s — too slow to feel live.
LOCAL_STT_MODEL = "Systran/faster-whisper-medium"
LOCAL_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
LOCAL_TTS_VOICE = "af_bella"


# ---------------------------------------------------------------------------
# Custom TTS class for Speaches/Kokoro.
# LiveKit's openai.TTS plugin doesn't correctly yield audio bytes from
# speaches' /v1/audio/speech response — symptom is "no audio frames pushed"
# errors. This class uses plain httpx POST with response_format="pcm" so the
# AudioEmitter treats bytes as raw 24kHz mono 16-bit PCM with no decoding.
# ---------------------------------------------------------------------------
class _SpeachesTTS(livekit_tts.TTS):
    def __init__(
        self, *, base_url: str, model: str, voice: str, api_key: str = "speaches"
    ) -> None:
        super().__init__(
            capabilities=livekit_tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._api_key = api_key

    def synthesize(self, text: str, *, conn_options: APIConnectOptions | None = None):
        return _SpeachesTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )


class _SpeachesTTSStream(livekit_tts.ChunkedStream):
    async def _run(self, output_emitter: livekit_tts.AudioEmitter) -> None:
        tts: _SpeachesTTS = self._tts  # type: ignore
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{tts._base_url}/audio/speech",
                json={
                    "model": tts._model,
                    "input": self._input_text,
                    "voice": tts._voice,
                    "response_format": "pcm",
                },
                headers={"Authorization": f"Bearer {tts._api_key}"},
            )
            r.raise_for_status()
            data = r.content

        output_emitter.initialize(
            request_id="speaches",
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        CHUNK = 4096
        for i in range(0, len(data), CHUNK):
            output_emitter.push(data[i:i + CHUNK])
        output_emitter.flush()


# ---------------------------------------------------------------------------
# Per-slot plugin builders. Each takes a choice string ("cloud" or "local")
# and returns a configured LiveKit STT/LLM/TTS instance. Unknown choices
# fall back to cloud — keeps the live site working even if the frontend
# sends garbage metadata.
# ---------------------------------------------------------------------------
def build_stt(choice: str):
    if choice == "local":
        logger.info(f"stt: local ({LOCAL_STT_MODEL} @ Speaches VPS)")
        return openai.STT(
            base_url=LOCAL_SPEACHES_URL,
            api_key="speaches",
            model=LOCAL_STT_MODEL,
            language="en",
        )
    logger.info("stt: cloud (Deepgram nova-3)")
    return deepgram.STT(model="nova-3", language="multi")


def build_llm(choice: str):
    if choice == "local":
        logger.info("llm: local (aika-llm via Ollama VPS)")
        return openai.LLM(
            base_url=LOCAL_OLLAMA_URL,
            api_key="ollama",
            model=LOCAL_LLM_MODEL,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    logger.info("llm: cloud (Groq Llama-3.3-70B)")
    return groq.LLM(model="llama-3.3-70b-versatile")


def build_tts(choice: str):
    if choice == "local":
        logger.info("tts: local (Kokoro via Speaches VPS)")
        # StreamAdapter wraps the non-streaming SpeachesTTS so the LLM output
        # is fed sentence-by-sentence — first sentence starts speaking before
        # the LLM has finished generating.
        return livekit_tts.StreamAdapter(
            tts=_SpeachesTTS(
                base_url=LOCAL_SPEACHES_URL,
                model=LOCAL_TTS_MODEL,
                voice=LOCAL_TTS_VOICE,
            ),
        )
    logger.info("tts: cloud (Cartesia)")
    return cartesia.TTS(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")


# ---------------------------------------------------------------------------
# AIKA agent definition. Same persona, same tools — just shared between
# both deployment modes.
# ---------------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are AIKA, an AI Kitchen Assistant helping chefs in a restaurant.
            You respond via voice so keep responses short and clear. No formatting, no emojis.

            Your main job is managing timers. When a chef says something like "timer steak 4 minutes" or "pasta 3 minutes", use the set_timer tool.
            When they ask "how long for steak" or "check timers", use the check_timers tool.
            When they say "cancel steak timer", use the cancel_timer tool.

            Keep confirmations brief, like "Timer set. Steak, 4 minutes." or "Pasta has 2 minutes left."
            Be helpful but stay out of the way. Chefs are busy.""",
        )
        self.timers: dict = {}

    @function_tool
    async def set_timer(self, context: RunContext, name: str, minutes: float):
        """Set a kitchen timer that will announce when done.

        Args:
            name: What the timer is for, like "steak" or "pasta"
            minutes: How many minutes to set the timer for
        """
        if name.lower() in self.timers:
            self.timers[name.lower()]["task"].cancel()

        end_time = time.time() + (minutes * 60)

        async def timer_callback():
            await asyncio.sleep(minutes * 60)
            del self.timers[name.lower()]
            session = context.session
            await session.say(f"{name} is ready!")
            logger.info(f"Timer finished: {name}")

        task = asyncio.create_task(timer_callback())
        self.timers[name.lower()] = {"end_time": end_time, "task": task}

        logger.info(f"Timer set: {name} for {minutes} minutes")
        m = int(minutes) if minutes == int(minutes) else minutes
        unit = "minute" if m == 1 else "minutes"
        await context.session.say(f"{name} timer set, {m} {unit}.")
        raise StopResponse()

    @function_tool
    async def check_timers(self, context: RunContext):
        """Check all active timers and how much time is left on each."""
        if not self.timers:
            reply = "No active timers."
        else:
            parts = []
            now = time.time()
            for name, timer in self.timers.items():
                remaining = max(0, timer["end_time"] - now)
                if remaining >= 60:
                    mins = int(remaining // 60)
                    unit = "minute" if mins == 1 else "minutes"
                    parts.append(f"{name} has {mins} {unit} left")
                else:
                    secs = int(remaining)
                    unit = "second" if secs == 1 else "seconds"
                    parts.append(f"{name} has {secs} {unit} left")
            reply = ". ".join(parts)
        await context.session.say(reply)
        raise StopResponse()

    @function_tool
    async def cancel_timer(self, context: RunContext, name: str):
        """Cancel an active timer.

        Args:
            name: The name of the timer to cancel, like "steak" or "pasta"
        """
        if name.lower() in self.timers:
            self.timers[name.lower()]["task"].cancel()
            del self.timers[name.lower()]
            logger.info(f"Timer cancelled: {name}")
            await context.session.say(f"{name} cancelled.")
        else:
            await context.session.say(f"No timer for {name}.")
        raise StopResponse()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Read the per-call stack metadata. Frontend posts this to /api/token
    # which embeds it into RoomConfiguration.agents[0].metadata.
    raw = ctx.job.metadata or "{}"
    try:
        choices = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"could not parse job metadata: {raw!r} — defaulting to cloud")
        choices = {}

    stt_choice = choices.get("stt", "cloud")
    llm_choice = choices.get("llm", "cloud")
    tts_choice = choices.get("tts", "cloud")
    logger.info(f"📋 stack stt={stt_choice} llm={llm_choice} tts={tts_choice}")

    # Local inference is slow — give the framework a longer timeout for any
    # slot that's pointing at a self-hosted endpoint, otherwise it'll retry
    # mid-generation and pile up parallel CPU work.
    long = APIConnectOptions(max_retry=1, retry_interval=2.0, timeout=90.0)

    # Only apply the long timeout to slots that are actually local — cloud
    # APIs use the default and we want their defaults so we don't regress
    # the working cloud-only path.
    has_local = "local" in (stt_choice, llm_choice, tts_choice)
    session_kwargs: dict = {
        "stt": build_stt(stt_choice),
        "llm": build_llm(llm_choice),
        "tts": build_tts(tts_choice),
        "vad": ctx.proc.userdata["vad"],
    }
    if has_local:
        session_kwargs["conn_options"] = SessionConnectOptions(
            stt_conn_options=long if stt_choice == "local" else DEFAULT_API_CONNECT_OPTIONS,
            llm_conn_options=long if llm_choice == "local" else DEFAULT_API_CONNECT_OPTIONS,
            tts_conn_options=long if tts_choice == "local" else DEFAULT_API_CONNECT_OPTIONS,
        )

    session = AgentSession(**session_kwargs)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

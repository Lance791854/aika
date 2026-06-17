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
)
from livekit.agents import (
    tts as livekit_tts,
)
from livekit.agents.llm import StopResponse
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import cartesia, deepgram, groq, openai, silero

import storage

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
            output_emitter.push(data[i : i + CHUNK])
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
    # keyterm prompting stops nova-3 from dropping the unfamiliar name "AIKA"
    # (esp. as the first word of an utterance). It's nova-3 + English only, so
    # we pin language="en" — fine for the kitchen demo. Drop keyterm and switch
    # back to "multi" if multilingual recognition is ever needed.
    return deepgram.STT(model="nova-3", language="en", keyterm=["AIKA", "Aika"])


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
    WAKE_WINDOW_SEC = 30

    # Words that count as "addressing AIKA" — STTs often mishear the name
    # depending on accent, so we include common phonetic neighbours seen in
    # both Deepgram and Whisper outputs. Compared lowercase.
    WAKE_WORDS = (
        "aika",
        "ika",
        "ayka",
        "ica",  # the basics
        "alka",
        "alika",
        "iika",  # Deepgram variants
        "aica",
        "aiko",
        "ayko",  # ending-vowel variants
    )
    # How long AIKA stays "awake" after a wake word in `window` mode. Lets the
    # user say "AIKA" once and then chat normally — works around STT dropping
    # the wake word on follow-up utterances.
    WAKE_WINDOW_SEC = 30.0

    def __init__(self, wake_mode: str = "off", publish_fn=None) -> None:
        super().__init__(
            instructions="""You are AIKA, an AI Kitchen Assistant helping chefs in a restaurant.
            You respond via voice so keep responses short and clear. No formatting, no emojis.

            You manage timers, temperatures, and notes.

            Timers. When a chef says something like "timer steak 4 minutes" or "pasta 3 minutes", use the set_timer tool.
            When they ask "how long for steak" or "check timers", use the check_timers tool.
            When they say "cancel steak timer", use the cancel_timer tool.

            Temperatures. When a chef says something like "log the freezer at minus eighteen" or "fridge is four degrees", use the log_temperature tool.
            Locations are things like "freezer", "fridge", "chicken", "lamb". Negative numbers are fine for frozen items.
            When they ask "what was the freezer at" or "check the fridge temperature", use the check_temperature tool.

            Notes. When a chef says something like "make a note we're out of butter" or "remind the morning shift to defrost the lamb", use the add_note tool. This is for anything they want remembered later — out-of-stock items, handovers, reminders.
            When they ask "what notes do we have", "what are we out of", or "what did I say to do", use the list_notes tool.

            Keep confirmations brief, like "Timer set. Steak, 4 minutes.", "Logged. Freezer at minus 18.", or "Noted."
            Be helpful but stay out of the way. Chefs are busy.""",
        )
        self.timers: dict = {}
        self._wake_mode = wake_mode
        self._publish = publish_fn or (lambda p: None)
        self._last_wake_at: float = 0

    def _emit_state(self) -> None:
        temps = []
        for t in storage.recent_temperatures(5):
            temps.append(
                {
                    **t,
                    "out_of_range": storage.check_range(t["location"], t["celsius"])
                    is not None,
                }
            )
        self._publish(
            {
                "type": "state",
                "timers": [
                    {"name": n, "end_time": t["end_time"]}
                    for n, t in self.timers.items()
                ],
                "temperatures": temps,
                "notes": storage.recent_notes(5),
            }
        )

    async def on_user_turn_completed(self, chat_ctx, new_message):
        if self._wake_mode == "off":
            return
        text = (new_message.text_content or "").lower()
        heard_wake = any(w in text for w in self.WAKE_WORDS)
        now = time.time()
        if heard_wake:
            self._last_wake_at = now
            return
        if (
            self._wake_mode == "window"
            and (now - self._last_wake_at) < self.WAKE_WINDOW_SEC
        ):
            return
        logger.info(f"wake gate ({self._wake_mode}) skipping reply for {text!r}")
        raise StopResponse()

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
        self._emit_state()
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
            self._emit_state()
        else:
            await context.session.say(f"No timer for {name}.")
        raise StopResponse()

    @function_tool
    async def log_temperature(self, context: RunContext, location: str, celsius: float):
        """Record a temperature reading for a fridge, freezer, or cooked item.

        Args:
            location: Where the reading is from, like "freezer", "fridge", "chicken"
            celsius: The temperature in degrees Celsius (use negatives for sub-zero)
        """
        storage.add_temperature(location, celsius)
        logger.info(f"logged temperature: {location} = {celsius}C")
        c = int(celsius) if celsius == int(celsius) else celsius
        reply = f"Logged. {location} at {c} degrees."
        warning = storage.check_range(location, celsius)
        if warning:
            reply += f" Warning. {warning.capitalize()}."
        await context.session.say(reply)
        self._emit_state()
        raise StopResponse()

    @function_tool
    async def check_temperature(self, context: RunContext, location: str):
        """Read back the most recent temperature for a location.

        Args:
            location: Where to check, like "freezer" or "fridge"
        """
        entry = storage.latest_temperature(location)
        if not entry:
            await context.session.say(f"No reading for {location}.")
        else:
            c = entry["celsius"]
            c = int(c) if c == int(c) else c
            await context.session.say(f"{location} was {c} degrees.")
        raise StopResponse()

    @function_tool
    async def add_note(self, context: RunContext, text: str):
        """Save a short note for the kitchen — out-of-stock items, reminders, handovers.

        Args:
            text: What to remember, like "out of butter" or "defrost lamb for tomorrow"
        """
        storage.add_note(text)
        logger.info(f"note added: {text}")
        await context.session.say("Noted.")
        self._emit_state()
        raise StopResponse()

    @function_tool
    async def list_notes(self, context: RunContext):
        """Read back saved notes — used for "what notes", "what are we out of", "what did I say to do"."""
        notes = storage.list_notes()
        if not notes:
            await context.session.say("No notes yet.")
        else:
            # last few only — chefs don't want a wall of text
            recent = [n["text"] for n in notes[-5:]]
            await context.session.say(". ".join(recent))
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
    # wake mode: "off" (always respond), "window" (wake once, awake 30s),
    # or "strict" (every utterance needs the wake word).
    wake_mode = str(choices.get("wake", "off"))
    if wake_mode not in ("off", "window", "strict"):
        wake_mode = "off"
    logger.info(
        f"📋 stack stt={stt_choice} llm={llm_choice} tts={tts_choice}"
        f" wake_mode={wake_mode}"
    )

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
            stt_conn_options=long
            if stt_choice == "local"
            else DEFAULT_API_CONNECT_OPTIONS,
            llm_conn_options=long
            if llm_choice == "local"
            else DEFAULT_API_CONNECT_OPTIONS,
            tts_conn_options=long
            if tts_choice == "local"
            else DEFAULT_API_CONNECT_OPTIONS,
        )

    session = AgentSession(**session_kwargs)

    # Publish events over the LiveKit data channel so the frontend debug
    # overlay can show what's actually happening (transcripts, tool calls,
    # reply text, timings). Frontend filters by topic="aika-debug".
    def _publish(payload: dict) -> None:
        payload["at"] = time.time()

        async def _send():
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps(payload).encode(),
                    topic="aika-debug",
                    reliable=True,
                )
            except Exception as e:
                logger.warning(f"publish_data failed: {e}")

        asyncio.create_task(_send())

    def _on_metrics(ev):
        m = ev.metrics
        t = m.type
        if t == "stt_metrics":
            _publish(
                {
                    "type": "stt_metrics",
                    "duration": m.duration,
                    "audio_duration": m.audio_duration,
                }
            )
        elif t == "eou_metrics":
            _publish(
                {"type": "eou_metrics", "transcription_delay": m.transcription_delay}
            )
        elif t == "llm_metrics":
            _publish(
                {
                    "type": "llm_metrics",
                    "duration": m.duration,
                    "ttft": m.ttft,
                    "tokens": m.completion_tokens,
                    "tok_per_sec": m.tokens_per_second,
                }
            )
        elif t == "tts_metrics":
            _publish(
                {
                    "type": "tts_metrics",
                    "duration": m.duration,
                    "ttfb": m.ttfb,
                    "chars": m.characters_count,
                }
            )

    def _on_user_transcribed(ev):
        if ev.is_final:
            _publish({"type": "user_text", "text": ev.transcript})

    def _on_item_added(ev):
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None) or ""
        if role == "assistant" and text.strip():
            _publish({"type": "assistant_text", "text": text})

    def _on_tools_executed(ev):
        for fc in ev.function_calls:
            args = fc.arguments
            # Arguments come through as a JSON string from the LLM — parse for
            # nicer rendering, fall back to raw if it isn't valid JSON.
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
            _publish({"type": "tool_call", "name": fc.name, "args": args})

    session.on("metrics_collected", _on_metrics)
    session.on("user_input_transcribed", _on_user_transcribed)
    session.on("conversation_item_added", _on_item_added)
    session.on("function_tools_executed", _on_tools_executed)

    # Publish stack on join so the overlay can show what's active.
    _publish({"type": "stack", "stt": stt_choice, "llm": llm_choice, "tts": tts_choice})

    assistant = Assistant(wake_mode=wake_mode, publish_fn=_publish)
    await session.start(agent=assistant, room=ctx.room)
    # initial state dump so the overlay shows what's already saved.
    assistant._emit_state()

    # frontend buttons publish to this topic — agent listens and acts.
    async def run_safety_check():
        bad = []
        seen = set()
        for entry in reversed(storage._load()["temperatures"]):
            loc = entry["location"]
            if loc in seen:
                continue
            seen.add(loc)
            w = storage.check_range(loc, entry["celsius"])
            if w:
                bad.append(w)
        if bad:
            await session.say("Warning. " + ". ".join(bad) + ".")
        else:
            await session.say("All temperatures within safe range.")

    async def inject_test_reading(location: str, severity: str):
        # find what counts as low / ok / high for this location and log it.
        rng = storage.TEMP_RANGES.get(location.lower())
        if not rng:
            return
        low, high = rng
        if severity == "low":
            value = low - 5
        elif severity == "high":
            value = high + 5
        else:
            value = (low + high) / 2
        storage.add_temperature(location, value)
        assistant._emit_state()

    def _on_data(data_packet):
        try:
            payload = json.loads(data_packet.data.decode())
        except Exception:
            return
        t = payload.get("type")
        if t == "check_temps":
            asyncio.create_task(run_safety_check())
        elif t == "inject_temp":
            asyncio.create_task(
                inject_test_reading(
                    payload.get("location", ""), payload.get("severity", "ok")
                )
            )

    ctx.room.on("data_received", _on_data)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

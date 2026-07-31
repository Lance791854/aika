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
import os
import re
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
    stt as livekit_stt,
)
from livekit.agents import (
    tts as livekit_tts,
)
from livekit import rtc
from livekit.agents.llm import StopResponse
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
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

# GPU stack (RunPod). Parakeet STT served by inference/parakeet/ on the pod,
# reached over the VPS->pod SSH tunnel (pm2 "aika-gpu-tunnel") on localhost:9000.
GPU_STT_URL = os.getenv("AIKA_GPU_STT_URL", "http://localhost:9000/v1")
GPU_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
# Qwen3 (Ollama) and Kokoro-FastAPI on the same pod, over the same tunnel.
# 11435 not 11434 locally so nothing collides with a CPU-stack dev tunnel.
GPU_LLM_URL = os.getenv("AIKA_GPU_LLM_URL", "http://localhost:11435/v1")
GPU_LLM_MODEL = "qwen3:8b"
GPU_TTS_URL = os.getenv("AIKA_GPU_TTS_URL", "http://localhost:8880/v1")
GPU_TTS_MODEL = "kokoro"
GPU_TTS_VOICE = "af_bella"  # same voice as the CPU stack, for clean A/B

# Cloudflare Workers AI. Credentials come from .env.local, never the repo.
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"
# same model Groq serves — lets us compare providers on equal footing
CF_LLM_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


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
    if choice == "gpu":
        logger.info(f"stt: gpu ({GPU_STT_MODEL} @ Parakeet/RunPod)")
        return openai.STT(
            base_url=GPU_STT_URL,
            api_key="parakeet",
            model=GPU_STT_MODEL,
            language="en",
        )
    if choice == "local":
        logger.info(f"stt: local ({LOCAL_STT_MODEL} @ Speaches VPS)")
        return openai.STT(
            base_url=LOCAL_SPEACHES_URL,
            api_key="speaches",
            model=LOCAL_STT_MODEL,
            language="en",
        )
    if choice == "cartesia":
        logger.info("stt: cartesia (ink-whisper)")
        # Cartesia is cloud-only (hosted). Prefer a dedicated STT key if set,
        # else fall back to the shared CARTESIA_API_KEY the TTS already uses.
        return cartesia.STT(
            model="ink-whisper",
            language="en",
            api_key=os.getenv("CARTESIA_STT_API_KEY") or None,
        )
    logger.info("stt: cloud (Deepgram nova-3)")
    # keyterm prompting stops nova-3 from dropping the unfamiliar name "AIKA"
    # (esp. as the first word of an utterance). It's nova-3 + English only, so
    # we pin language="en" — fine for the kitchen demo. Drop keyterm and switch
    # back to "multi" if multilingual recognition is ever needed.
    return deepgram.STT(model="nova-3", language="en", keyterm=["AIKA", "Aika"])


def build_llm(choice: str):
    if choice == "cf":
        if CF_ACCOUNT_ID and CF_API_TOKEN:
            logger.info(f"llm: cloudflare ({CF_LLM_MODEL})")
            return openai.LLM(
                base_url=CF_BASE_URL,
                api_key=CF_API_TOKEN,
                model=CF_LLM_MODEL,
            )
        logger.warning("cf llm selected but CF credentials missing — using cloud")
    if choice == "gpu":
        logger.info(f"llm: gpu ({GPU_LLM_MODEL} @ Ollama/RunPod)")
        # reasoning_effort="none" disables qwen3 thinking mode — with it on,
        # the model burns seconds of reasoning tokens before the first word.
        return openai.LLM(
            base_url=GPU_LLM_URL,
            api_key="ollama",
            model=GPU_LLM_MODEL,
            reasoning_effort="none",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
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
    if choice == "gpu":
        logger.info("tts: gpu (Kokoro-FastAPI @ RunPod)")
        # Kokoro-FastAPI speaks the same /v1/audio/speech API as speaches,
        # so the pcm-based workaround class works unchanged.
        return livekit_tts.StreamAdapter(
            tts=_SpeachesTTS(
                base_url=GPU_TTS_URL,
                model=GPU_TTS_MODEL,
                voice=GPU_TTS_VOICE,
                api_key="kokoro",
            ),
        )
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
# Compare STT: run several engines on the same audio, publish all transcripts
# to the debug overlay, and return the first (primary) result for the agent to
# act on. Used by the debug "compare STT" toggle to eyeball Parakeet vs cloud.
# ---------------------------------------------------------------------------
class _CompareSTT(livekit_stt.STT):
    def __init__(self, engines: list[tuple[str, livekit_stt.STT]], publish) -> None:
        super().__init__(capabilities=engines[0][1].capabilities)
        self._engines = engines
        self._publish = publish

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        async def safe(engine):
            t0 = time.time()
            try:
                ev = await engine.recognize(
                    buffer, language=language, conn_options=conn_options
                )
                return ev, None, (time.time() - t0) * 1000
            except Exception as e:
                return None, e, (time.time() - t0) * 1000

        results = await asyncio.gather(*[safe(eng) for _, eng in self._engines])

        def text_of(ev, err):
            if err:
                return f"<err: {type(err).__name__}>"
            if not ev or not ev.alternatives:
                return "<empty>"
            return ev.alternatives[0].text

        self._publish(
            {
                "type": "stt_compare",
                "results": [
                    {"label": label, "text": text_of(ev, err), "ms": round(ms)}
                    for (label, _), (ev, err, ms) in zip(self._engines, results)
                ],
            }
        )

        # Return the first engine's result (the primary the agent acts on).
        for _, (ev, _err, _ms) in zip(self._engines, results):
            if ev is not None:
                return ev
        raise results[0][1]


# ---------------------------------------------------------------------------
# "device" wake mode. Works like the planned ESP32 wearable: whisper-tiny on
# the CPU box checks each utterance for "AIKA" first. Only after that does
# audio go to the real STT and LLM. Everything else never leaves our servers.
# ---------------------------------------------------------------------------
WAKE_SPOT_MODEL = "Systran/faster-whisper-tiny.en"
WAKE_SPOT_WINDOW_S = 10.0


def device_gate(spot_text: str, armed_until: float, now: float) -> tuple[bool, float]:
    """Forward/drop decision. Returns (forward, new_armed_until).

    Same rules as the strict wake mode: a full command is handled and the mic
    closes again. Only a bare "AIKA" leaves a 10s window open, and one
    follow-up uses it up.
    """
    if now < armed_until:
        return True, 0.0
    text = spot_text.lower()
    # also catch spelled-out forms like "A.I.K.A." — squash to letters only
    compact = re.sub(r"[^a-z]", "", text)
    if Assistant.WAKE_RE.search(text) or "aika" in compact:
        rest = [
            w
            for w in re.findall(r"[a-z']+", Assistant.WAKE_RE.sub(" ", text))
            if w not in Assistant._FILLER
        ]
        if not rest:
            return True, now + WAKE_SPOT_WINDOW_S
        return True, 0.0
    return False, armed_until


class _WakeGatedSTT(livekit_stt.STT):
    def __init__(self, inner: livekit_stt.STT, publish) -> None:
        # non-streaming on purpose, even when the inner engine streams —
        # the session then VAD-chunks audio and every chunk passes the wake
        # check before anything is sent on.
        super().__init__(
            capabilities=livekit_stt.STTCapabilities(
                streaming=False, interim_results=False
            )
        )
        self._inner = inner
        self._publish = publish
        self._armed_until = 0.0

    async def _spot(self, buffer: AudioBuffer) -> str:
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"{LOCAL_SPEACHES_URL}/audio/transcriptions",
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data={
                        "model": WAKE_SPOT_MODEL,
                        "language": "en",
                        # the model has never seen the name — this hint makes
                        # it write "AIKA" instead of "I can" or "A.I.K.A."
                        "prompt": (
                            "The kitchen assistant is called AIKA. "
                            "Chefs say things like: AIKA, set a timer. hey AIKA."
                        ),
                    },
                    headers={"Authorization": "Bearer speaches"},
                )
                r.raise_for_status()
                return r.json().get("text", "")
        except Exception as e:
            logger.warning(f"wake check server unreachable, staying silent: {e}")
            return ""

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        now = time.time()
        # inside the window: forward straight away, no spotting needed
        forward, self._armed_until = device_gate("", self._armed_until, now)
        spot = ""
        if not forward:
            spot = await self._spot(buffer)
            forward, self._armed_until = device_gate(spot, self._armed_until, now)
        if forward:
            self._publish({"type": "wake_gate", "forwarded": True, "spot": spot})
            return await self._inner.recognize(
                buffer, language=language, conn_options=conn_options
            )
        self._publish({"type": "wake_gate", "forwarded": False, "spot": spot})
        logger.info(f"wake gate dropped utterance: {spot!r}")
        return livekit_stt.SpeechEvent(
            type=livekit_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[livekit_stt.SpeechData(language="en", text="")],
        )


# ---------------------------------------------------------------------------
# AIKA agent definition. Same persona, same tools — just shared between
# both deployment modes.
# ---------------------------------------------------------------------------
class Assistant(Agent):
    # Words that count as "addressing AIKA" — STTs often mishear the name
    # depending on accent, so we include common phonetic neighbours seen in
    # both Deepgram and Whisper outputs.
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
    # Match as whole words only. Substring matching let fragments like "ica"
    # and "ika" fire on ordinary speech — "basically", "medical", "paprika",
    # "America" — so the gate woke on side conversation and replied anyway.
    WAKE_RE = re.compile(r"\b(?:" + "|".join(WAKE_WORDS) + r")\b")
    # A bare address ("AIKA." then a pause) opens the gate this long for the
    # follow-up turn — STT splits the pause into two turns and the command
    # half has no wake word in it.
    WAKE_WINDOW_S = 10.0
    _FILLER = {"hey", "ok", "okay", "um", "uh"}

    def __init__(self, wake_mode: str = "off", publish_fn=None) -> None:
        super().__init__(
            instructions="""You are AIKA, an AI Kitchen Assistant helping chefs in a restaurant.
            You respond via voice so keep responses short and clear. No formatting, no emojis.

            You manage timers, reminders, temperatures, and notes.

            Timers. When a chef says something like "timer steak 4 minutes" or "pasta 3 minutes", use the set_timer tool.
            When they ask "how long for steak" or "check timers", use the check_timers tool.
            When they say "cancel steak timer", use the cancel_timer tool.

            Reminders. When a chef gives a task WITH a time delay, like "remind me in 20 minutes to rotate the stock", use the set_reminder tool. You will speak the reminder aloud when the time is up.
            When they ask "what reminders are set", use the check_reminders tool. When they say "cancel the stock reminder", use the cancel_reminder tool.
            If no time is given — like "remind the morning shift to defrost the lamb" — that is a note, not a reminder. Use add_note and never invent a delay.

            Temperatures. When a chef says something like "log the freezer at minus eighteen" or "fridge is four degrees", use the log_temperature tool.
            Locations are things like "freezer", "fridge", "chicken", "lamb". Negative numbers are fine for frozen items.
            When they ask "what was the freezer at" or "check the fridge temperature", use the check_temperature tool.

            Notes. When a chef says something like "make a note we're out of butter" or "remind the morning shift to defrost the lamb", use the add_note tool. This is for anything they want remembered later — out-of-stock items, handovers, messages for other shifts. If they give a specific time delay, that is a reminder, not a note — use set_reminder.
            When they ask "what notes do we have", "what are we out of", or "what did I say to do", use the list_notes tool.

            Shift summary. When a chef asks for "the shift summary", "end of shift report", or "what happened this shift", use the shift_summary tool.

            Unhandled requests. If a chef asks you to DO something that is not a timer, temperature, or note — like placing a supplier order, controlling equipment, playing music, or looking up a recipe — call the log_unhandled tool with a short summary of what they asked, then briefly tell them you can't do that. Do NOT call it for greetings, small talk, thanks, or anything you can just answer in conversation.

            Keep confirmations brief, like "Timer set. Steak, 4 minutes.", "Logged. Freezer at minus 18.", or "Noted."
            Be helpful but stay out of the way. Chefs are busy.""",
        )
        self.timers: dict = {}
        self.reminders: dict = {}
        self._wake_mode = wake_mode
        self._wake_open_until = 0.0
        self._publish = publish_fn or (lambda p: None)

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
                "reminders": [
                    {"text": t, "end_time": r["end_time"]}
                    for t, r in self.reminders.items()
                ],
                "temperatures": temps,
                "notes": storage.recent_notes(5),
                "unhandled": storage.recent_unhandled(5),
            }
        )

    async def on_user_turn_completed(self, chat_ctx, new_message):
        # "off" replies to everything; "strict" only replies to utterances that
        # address AIKA by name (e.g. "AIKA ..." or "hey AIKA ...").
        if self._wake_mode == "off":
            return
        if self._wake_mode == "device":
            # gating happened pre-STT in _WakeGatedSTT — only the empty turns
            # the gate left behind need dropping here.
            if not (new_message.text_content or "").strip():
                raise StopResponse()
            return
        text = (new_message.text_content or "").lower()
        if self.WAKE_RE.search(text):
            # A bare address (nothing left after wake words and filler) opens
            # a short window so the follow-up turn gets through. A turn that
            # already carries a command does not — side chatter stays gated.
            rest = [
                w
                for w in re.findall(r"[a-z']+", self.WAKE_RE.sub(" ", text))
                if w not in self._FILLER
            ]
            if not rest:
                self._wake_open_until = time.time() + self.WAKE_WINDOW_S
            return
        if time.time() < self._wake_open_until:
            self._wake_open_until = 0.0  # window is for one follow-up turn
            return
        logger.info(f"wake gate skipping reply for {text!r}")
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
    async def set_reminder(self, context: RunContext, text: str, minutes: float):
        """Set a timed reminder that will be spoken aloud when the time is up.

        Only for requests with an explicit delay, like "remind me in 20 minutes".
        If the chef gave no time, use add_note instead — never guess the minutes.

        Args:
            text: What to remind about, like "rotate the stock"
            minutes: How many minutes from now to speak the reminder
        """
        key = text.lower()
        if key in self.reminders:
            self.reminders[key]["task"].cancel()

        end_time = time.time() + (minutes * 60)

        async def reminder_callback():
            await asyncio.sleep(minutes * 60)
            del self.reminders[key]
            await context.session.say(f"Reminder. {text}.")
            logger.info(f"Reminder fired: {text}")
            self._emit_state()

        task = asyncio.create_task(reminder_callback())
        self.reminders[key] = {"end_time": end_time, "task": task}

        logger.info(f"Reminder set: {text} in {minutes} minutes")
        self._emit_state()
        m = int(minutes) if minutes == int(minutes) else minutes
        unit = "minute" if m == 1 else "minutes"
        await context.session.say(f"Reminder set, {m} {unit}.")
        raise StopResponse()

    @function_tool
    async def check_reminders(self, context: RunContext):
        """Check all pending reminders and how long until each one."""
        if not self.reminders:
            reply = "No reminders set."
        else:
            parts = []
            now = time.time()
            for text, r in self.reminders.items():
                remaining = max(0, r["end_time"] - now)
                if remaining >= 60:
                    mins = int(remaining // 60)
                    unit = "minute" if mins == 1 else "minutes"
                    parts.append(f"{text} in {mins} {unit}")
                else:
                    secs = int(remaining)
                    unit = "second" if secs == 1 else "seconds"
                    parts.append(f"{text} in {secs} {unit}")
            reply = "Reminders. " + ". ".join(parts) + "."
        await context.session.say(reply)
        raise StopResponse()

    @function_tool
    async def cancel_reminder(self, context: RunContext, keyword: str):
        """Cancel a pending reminder that matches a keyword.

        Args:
            keyword: A word from the reminder to cancel, like "stock"
        """
        for text, r in list(self.reminders.items()):
            if keyword.lower() in text:
                r["task"].cancel()
                del self.reminders[text]
                logger.info(f"Reminder cancelled: {text}")
                await context.session.say(f"Cancelled the {keyword} reminder.")
                self._emit_state()
                break
        else:
            await context.session.say(f"No reminder about {keyword}.")
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

    @function_tool
    async def shift_summary(self, context: RunContext):
        """Read back the end-of-shift summary — temperatures logged, notes, and
        anything AIKA couldn't handle this shift."""
        await context.session.say(storage.shift_summary())
        raise StopResponse()

    @function_tool
    async def log_unhandled(self, context: RunContext, request: str):
        """Record a request AIKA can't fulfil (not a timer, temperature, or note),
        so the coverage gap can be reviewed later.

        Args:
            request: A short summary of what the chef asked for, like "order more beef" or "turn on the oven"
        """
        storage.add_unhandled(request)
        logger.info(f"unhandled request: {request}")
        await context.session.say("Sorry, I can't help with that one.")
        self._emit_state()
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
    # wake mode: "off" (always respond) or "strict" (only respond when the
    # utterance addresses AIKA by name). "window" is the old 3-mode frontend's
    # value — fold it into strict so a not-yet-updated frontend still gates.
    wake_mode = str(choices.get("wake", "off"))
    if wake_mode == "window":
        wake_mode = "strict"
    if wake_mode not in ("off", "strict", "device"):
        wake_mode = "off"
    logger.info(
        f"📋 stack stt={stt_choice} llm={llm_choice} tts={tts_choice}"
        f" wake_mode={wake_mode}"
    )

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

    # Debug "compare STT" toggle: run Parakeet (primary, what the agent acts on)
    # alongside Deepgram on the same audio and publish both transcripts to the
    # overlay. Lets you see where Parakeet differs from cloud STT live.
    compare = bool(choices.get("compare", False))
    if compare:
        logger.info("stt: compare (Parakeet primary + Deepgram shadow)")
        stt_impl = _CompareSTT(
            [("Parakeet", build_stt("gpu")), ("Deepgram", build_stt("cloud"))],
            _publish,
        )
    else:
        stt_impl = build_stt(stt_choice)

    # wearable sim: check for "AIKA" locally before any audio reaches real STT
    if wake_mode == "device":
        logger.info("wake: device sim (local wake check before STT)")
        stt_impl = _WakeGatedSTT(stt_impl, _publish)

    # Local inference is slow — give the framework a longer timeout for any
    # slot that's pointing at a self-hosted endpoint, otherwise it'll retry
    # mid-generation and pile up parallel CPU work.
    long = APIConnectOptions(max_retry=1, retry_interval=2.0, timeout=90.0)

    # Only apply the long timeout to slots that are actually local — cloud
    # APIs use the default and we want their defaults so we don't regress
    # the working cloud-only path.
    has_local = "local" in (stt_choice, llm_choice, tts_choice)
    session_kwargs: dict = {
        "stt": stt_impl,
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
        elif t == "clear_unhandled":
            storage.clear_unhandled()
            assistant._emit_state()
        elif t == "delete_unhandled":
            storage.delete_unhandled(payload.get("at"))
            assistant._emit_state()
        elif t == "clear_notes":
            storage.clear_notes()
            assistant._emit_state()
        elif t == "delete_note":
            storage.delete_note(payload.get("at"))
            assistant._emit_state()
        elif t == "clear_temps":
            storage.clear_temperatures()
            assistant._emit_state()
        elif t == "delete_temp":
            storage.delete_temperature_location(payload.get("location", ""))
            assistant._emit_state()

    ctx.room.on("data_received", _on_data)

    # nudge the pod's LLM as the call starts so the first real turn doesn't
    # pay any cold-start cost. keep_alive=-1 on the pod does the heavy lifting;
    # this covers the first call after a pod reboot.
    if llm_choice == "gpu":

        async def _warm_gpu_llm():
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    await client.post(
                        f"{GPU_LLM_URL}/chat/completions",
                        json={
                            "model": GPU_LLM_MODEL,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                        },
                    )
                logger.info("gpu llm warmed")
            except Exception as e:
                logger.warning(f"gpu llm warm-up failed: {e}")

        asyncio.create_task(_warm_gpu_llm())

    # unattended re-alerting: if a bad reading never gets a corrective one,
    # AIKA speaks up again on its own every ALERT_COOLDOWN until it does.
    # Alert times persist in storage so a rejoin doesn't re-warn early.

    async def temp_monitor():
        while True:
            await asyncio.sleep(30)
            due = storage.overdue_temperatures(storage.last_alerted())
            if not due:
                continue
            for d in due:
                storage.mark_alerted(d["location"])
            logger.info(f"re-alerting: {[d['location'] for d in due]}")
            try:
                await session.say(
                    "Warning. "
                    + ". ".join(d["warning"] for d in due)
                    + ". Still not corrected."
                )
            except RuntimeError:
                return

    monitor_task = asyncio.create_task(temp_monitor())

    async def _stop_monitor():
        monitor_task.cancel()

    ctx.add_shutdown_callback(_stop_monitor)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

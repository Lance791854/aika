import asyncio
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
    stt as livekit_stt,
    tts as livekit_tts,
)
from livekit.agents.llm import StopResponse
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, noise_cancellation, openai, silero

logger = logging.getLogger("agent_local")

load_dotenv(".env.local")


# Self-hosted endpoints reached over the SSH tunnel:
#   ssh -L 11434:localhost:11434 -L 8000:localhost:8000 -N debian@<inference-host>
OLLAMA_URL = "http://localhost:11434/v1"
SPEACHES_URL = "http://localhost:8000/v1"

# "aika-llm" is a custom Ollama model variant created on the inference server:
#   FROM qwen2.5:7b
#   PARAMETER num_thread 8     # sweet spot on this CPU (8 cores = ~6 tok/s)
#   PARAMETER num_ctx 4096
# Other available models we benchmarked: qwen2.5:7b (raw, ~5 tok/s),
# gemma3:12b-it-qat (smarter but ~1.6 tok/s — too slow for voice on CPU).
# Set True to skip LLM + TTS — only log STT comparison results. Lets you
# rapid-fire utterances to compare STT models without waiting for replies.
STT_ONLY_MODE = True

LLM_MODEL = "aika-llm"
# faster-whisper-large-v3 (multilingual, full size). Slow on CPU (~10-15s
# per utterance) but the highest-accuracy Whisper available. Needed because
# the medium model misheard "steak" → "take" / "let's take" with an
# Australian accent — proven via side-by-side comparison with Deepgram.
STT_MODEL = "Systran/faster-whisper-large-v3"
TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
TTS_VOICE = "af_bella"


class CompareSTT(livekit_stt.STT):
    """Runs N STTs in parallel on the same audio and logs all results.
    Returns the first listed STT's result to the agent (the 'primary' for
    actual agent use). Useful for A/B/C/...-testing transcript quality across
    multiple local + cloud STT models on the same mic audio."""

    def __init__(self, engines: list[tuple[str, livekit_stt.STT]]) -> None:
        if not engines:
            raise ValueError("Need at least one STT engine")
        # Use the first engine's capabilities — they all need to be batch (streaming=False)
        super().__init__(capabilities=engines[0][1].capabilities)
        self._engines = engines
        self._label_width = max(len(lbl) for lbl, _ in engines)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        async def safe(engine):
            t0 = asyncio.get_event_loop().time()
            try:
                ev = await engine.recognize(buffer, language=language, conn_options=conn_options)
                return ev, None, asyncio.get_event_loop().time() - t0
            except Exception as e:
                return None, e, asyncio.get_event_loop().time() - t0

        results = await asyncio.gather(*[safe(eng) for _, eng in self._engines])

        def text_of(ev, err):
            if err:
                return f"<err: {type(err).__name__}>"
            if not ev or not ev.alternatives:
                return "<empty>"
            return ev.alternatives[0].text

        for (label, _), (ev, err, dt) in zip(self._engines, results):
            logger.info(
                f"🆎 {label:>{self._label_width}}: [{dt:5.2f}s] {text_of(ev, err)!r}"
            )

        # Return first non-error result; primary takes priority.
        for (_, _), (ev, err, _) in zip(self._engines, results):
            if ev is not None:
                return ev
        # All failed — re-raise the first error
        raise results[0][1]


class SpeachesTTS(livekit_tts.TTS):
    """Custom TTS for speaches — bypasses openai SDK which doesn't yield audio
    bytes from speaches' /v1/audio/speech response (cause unclear, possibly
    a streaming/content-type mismatch). Uses plain httpx POST + audio/pcm
    output so LiveKit's AudioEmitter treats bytes as raw 16-bit mono samples
    at 24kHz with no decoding."""

    def __init__(self, *, base_url: str, model: str, voice: str, api_key: str = "speaches"):
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
        from livekit.agents.tts.tts import DEFAULT_API_CONNECT_OPTIONS
        return _SpeachesStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )


class _SpeachesStream(livekit_tts.ChunkedStream):
    async def _run(self, output_emitter: livekit_tts.AudioEmitter) -> None:
        tts: SpeachesTTS = self._tts  # type: ignore
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{tts._base_url}/audio/speech",
                json={
                    "model": tts._model,
                    "input": self._input_text,
                    "voice": tts._voice,
                    "response_format": "pcm",  # raw 24kHz mono 16-bit
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
        # Push in ~85ms chunks for smoother playback start
        CHUNK = 4096
        for i in range(0, len(data), CHUNK):
            output_emitter.push(data[i:i + CHUNK])
        output_emitter.flush()


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
        self.timers = {}

    async def on_user_turn_completed(self, chat_ctx, new_message):
        # STT-only debug mode: log nothing further and skip LLM/TTS entirely.
        if STT_ONLY_MODE:
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
        # Speak the reply directly and skip the post-tool LLM call (saves ~5s).
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


async def _warm_llm_once() -> None:
    """Fire-and-forget call to make sure aika-llm is in RAM before the user speaks."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{OLLAMA_URL}/chat/completions",
                json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "."}],
                      "max_tokens": 1, "stream": False},
            )
    except Exception as e:
        logger.warning(f"warmup failed (non-fatal): {e}")


async def _keep_llm_warm_loop() -> None:
    """Ping aika-llm every 4 minutes so Ollama doesn't unload it mid-session."""
    while True:
        await asyncio.sleep(240)
        await _warm_llm_once()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Warm the LLM immediately + start a background keepalive ping every 4min.
    # Without this, an idle box will hit cold-start (~6s) on the first user
    # utterance. With it, the LLM stays in Ollama's RAM the whole session.
    warm_task = asyncio.create_task(_warm_llm_once())
    asyncio.create_task(_keep_llm_warm_loop())

    # LiveKit's default APIConnectOptions.timeout is 10s — too tight for CPU
    # inference with tool calls (qwen2.5:3b takes ~3-6s for a tool-calling
    # response on this box). The framework retries on timeout, which spawns
    # parallel calls that contend for CPU and make everything worse.
    long_timeout = APIConnectOptions(max_retry=1, retry_interval=2.0, timeout=90.0)

    # Per-stage timing for debugging. Each metrics event has duration/ttft.
    def _on_metrics(ev):
        m = ev.metrics
        t = m.type
        if t == "stt_metrics":
            logger.info(f"⏱  STT      {m.duration:6.2f}s  audio={m.audio_duration:.2f}s")
        elif t == "eou_metrics":
            logger.info(f"⏱  EOU      transcription_delay={m.transcription_delay:.2f}s")
        elif t == "llm_metrics":
            logger.info(
                f"⏱  LLM      {m.duration:6.2f}s  ttft={m.ttft:.2f}s  "
                f"out={m.completion_tokens}t  {m.tokens_per_second:.1f}t/s"
            )
        elif t == "tts_metrics":
            logger.info(
                f"⏱  TTS      {m.duration:6.2f}s  ttfb={m.ttfb:.2f}s  "
                f"audio={m.audio_duration:.2f}s  chars={m.characters_count}"
            )

    def _on_user_transcribed(ev):
        if ev.is_final:
            logger.info(f"🗣  user:    {ev.transcript!r}")

    def _on_item_added(ev):
        item = ev.item
        if getattr(item, "role", None) == "assistant":
            text = getattr(item, "text_content", None) or str(item)
            if text and text.strip():
                logger.info(f"🤖 aika:    {text!r}")

    def _on_tools_executed(ev):
        for fc in ev.function_calls:
            logger.info(f"🔧 tool:    {fc.name}({fc.arguments})")

    session = AgentSession(
        # Disable AEC warmup: with a headset there's no echo to cancel, and the
        # 3s "interruptions disabled" window is just dead time. Set to a small
        # value (or None) if you switch to laptop speakers + mic.
        aec_warmup_duration=None,
        conn_options=SessionConnectOptions(
            llm_conn_options=long_timeout,
            stt_conn_options=long_timeout,
            tts_conn_options=long_timeout,
        ),
        stt=CompareSTT([
            # First listed = primary (result used by agent). Add/remove freely.
            ("voxtral-3b (GPU)", openai.STT(
                base_url="http://localhost:8001/v1", api_key="local", language="en",
                model="mistralai/Voxtral-Mini-3B-2507")),
            ("deepgram", deepgram.STT(model="nova-3", language="multi")),
        ]),
        llm=openai.LLM(
            base_url=OLLAMA_URL,
            api_key="ollama",
            model=LLM_MODEL,
            # CPU inference can take >10s first-token; default timeout is too tight.
            timeout=httpx.Timeout(60.0, connect=10.0),
        ),
        # StreamAdapter wraps our non-streaming SpeachesTTS and feeds it the
        # LLM output sentence-by-sentence. First sentence's audio starts before
        # the LLM has finished generating — perceived latency goes down.
        tts=livekit_tts.StreamAdapter(
            tts=SpeachesTTS(
                base_url=SPEACHES_URL,
                model=TTS_MODEL,
                voice=TTS_VOICE,
            ),
        ),
        vad=ctx.proc.userdata["vad"],
    )

    session.on("metrics_collected", _on_metrics)
    session.on("user_input_transcribed", _on_user_transcribed)
    session.on("conversation_item_added", _on_item_added)
    session.on("function_tools_executed", _on_tools_executed)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # BVC: background voice cancellation. Filters non-human voices and
            # ambient noise out of the mic input. Helps with mishearing and
            # makes interruption detection more reliable.
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

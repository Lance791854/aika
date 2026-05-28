# AIKA — AI Kitchen Assistant

A voice AI agent for chefs in a busy kitchen, built with [LiveKit Agents](https://github.com/livekit/agents).
The chef talks, AIKA listens, sets timers, checks them, and speaks back unprompted when a timer fires.

University project exploring the **cost / privacy / latency tradeoff** between cloud APIs and
fully self-hosted inference on CPU-only hardware.

## Two variants

| File | Stack | Latency | Privacy |
|---|---|---|---|
| `src/agent.py` | LiveKit Cloud + Deepgram (STT) + Groq Llama-3.3-70B (LLM) + Cartesia (TTS) | ~1-2s | API providers see audio + text |
| `src/agent_local.py` | LiveKit transport only + self-hosted Ollama (qwen2.5:3b) + Speaches (Whisper STT + Kokoro TTS) | ~6-8s | nothing leaves your hardware |

Both share the same `Assistant` class — same prompt, same tools (`set_timer`, `check_timers`, `cancel_timer`).
The local variant is a drop-in replacement; only the plugin instantiations differ.

## Setup

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env.local   # fill in your LiveKit + provider keys
```

## Run

```bash
# Talk to AIKA in your terminal (uses mic + speakers directly)
uv run python src/agent.py console

# Or as a worker connected to a LiveKit room (for use with a web frontend)
uv run python src/agent.py dev

# Self-hosted variant — needs Ollama + Speaches reachable at the URLs in agent_local.py
uv run python src/agent_local.py dev
```

## Tests

```bash
uv run pytest
```

LLM-as-judge eval suite under `tests/` (friendliness, grounding, safety refusal).

## License

MIT

# AIKA

A voice AI agent for chefs. Built on [LiveKit Agents](https://github.com/livekit/agents).

The chef talks. AIKA listens, sets timers, logs temperatures, takes notes, and
speaks back unprompted when something needs attention.

## What it does

- **Timers** — `"AIKA, steak four minutes"` / `"how long for steak"` / `"cancel steak timer"`
- **Temperatures** — `"log the freezer at minus eighteen"` / `"what was the freezer at"`. Readings outside [FSANZ](https://www.foodstandards.gov.au/) safe ranges (freezer below -18, fridge 0-5, cooked poultry above 75, etc.) trigger an immediate warning. If no in range reading is logged, AIKA re-warns unprompted every 5 minutes until one is.
- **Reminders** — `"remind me in twenty minutes to rotate the stock"` / `"what reminders are set"` / `"cancel the stock reminder"`. Spoken aloud unprompted when the time is up. No time given means it's saved as a note instead.
- **Stock requests** — `"we're low on cream, order five kilos, urgent"` / `"what stock do we need"`. Shown in the panel and the shift summary.
- **Deleting** — `"delete the note about butter"`. AIKA reads it back and asks yes or no first. Works for notes and stock requests.
- **Notes** — `"make a note we're out of butter"` / `"what notes do we have"` / `"what are we out of"`
- **Shift summary** — `"give me the shift summary"`. Spoken end-of-shift report: latest temp per location (flagging anything out of range), notes, and a count of unhandled requests.
- **Wake modes** — always reply / only reply when addressed by name (`"AIKA ..."` or `"hey AIKA ..."`). Saying just `"AIKA"` and pausing opens a 10-second window for the follow-up command.
- **Wearable sim** — a wake mode that works like the planned ESP32 wearable. A small local model listens for `"AIKA"`. Until it hears it, nothing is sent to the STT or LLM at all.
- **Status panel** — fixed side panel during the call. Running timers with countdown, latest temperature per location, latest notes. Out-of-range temps go red. Includes a button to run a safety check (AIKA scans recent readings and speaks any warnings), plus per-card buttons to inject fake low/ok/high readings for testing the alerts.
- **Debug overlay** — per-turn STT / LLM / TTS timings, transcripts, tool calls
- **Per-call stack toggle** — pick cloud or self-hosted for each of STT / LLM / TTS independently. Defaults to cloud. Choices ride in the agent dispatch metadata.

## Layout

```
src/
  agent.py        Clean cloud-only reference. 3 timer tools. Forkable starting point.
  agent_local.py  Debug variant. CompareSTT lets you A/B multiple STT engines (incl Voxtral) on the same mic audio.
  agent_aika.py   Production agent. Reads ctx.job.metadata, picks plugins per call, all 7 tools, safety check listener.
  storage.py      JSON-file persistence for temperatures + notes. FSANZ ranges baked in.
frontend/         Next.js, forked from LiveKit's agent-starter-react. Toggles, status panel, debug overlay.
tests/            LLM-as-judge evals (friendliness, grounding, refusal).
```

## Architecture

```
  Browser
     │
     ▼
  Caddy + Next.js frontend
     │
     ▼
  LiveKit Cloud
     │
     ▼
  aika-worker (Python)
     │
     ├── Cloud APIs: Deepgram / Groq / Cartesia
     └── Self-hosted: Ollama + Speaches
```

The worker stays alive as a pm2 process. The frontend is a static `next start` behind Caddy
for HTTPS. The inference services are optional.

## Setup

Backend (agent + tests) needs Python 3.10+ and [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
cp .env.example .env.local
# Add LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
# and provider keys you actually want: DEEPGRAM_API_KEY, GROQ_API_KEY, CARTESIA_API_KEY
```

Frontend needs Node 20+ and [`pnpm`](https://pnpm.io/):

```bash
cd frontend
pnpm install
cp .env.example .env.local
# Same LIVEKIT_* keys as above
```

## Run

Three terminals for the full setup:

```bash
# 1. Production agent worker
uv run python src/agent_aika.py dev

# 2. Frontend dev server
cd frontend && pnpm dev

# 3. Optional: SSH tunnel to the self-hosted inference box,
#    so the "local" stack toggles actually reach Ollama + Speaches
ssh -L 11434:localhost:11434 -L 8000:localhost:8000 -N <user>@<inference-host>
```

Then open http://localhost:3000, pick a stack, click Start.

Just the cloud reference agent (no inference setup):

```bash
uv run python src/agent.py dev
```

STT-comparison variant for accent debugging:

```bash
uv run python src/agent_local.py dev
```

## Self-hosted inference (optional)

The "local" toggles route to:

- **Ollama** on `:11434` running `aika-llm` (qwen2.5:7b with `PARAMETER num_thread 8`)
- **Speaches** on `:8000` serving `Systran/faster-whisper-medium` (STT) and `speaches-ai/Kokoro-82M-v1.0-ONNX` (TTS)

Both can live on any machine the agent worker can reach. URLs are in `src/agent_aika.py`.

If the inference box is exposed to the public internet, lock the ports down to just the
agent worker's IP.

## Tests

```bash
uv run pytest
```

The behavior tests use Groq `llama-3.3-70b-versatile` as the judge, so they need `GROQ_API_KEY`.
Groq has a free tier, that is why we use it. The free tier has a small per minute limit, so the
suite pauses between tests and takes a few minutes. If you have OpenAI credits you can swap the
judge in `tests/` back to `gpt-4.1-mini`, it runs faster.

## License

MIT

# Parakeet STT server

OpenAI-compatible wrapper around NVIDIA `parakeet-tdt-0.6b-v2` (NeMo). Drop-in
replacement for the Speaches/Whisper STT endpoint the agent already uses —
the agent talks to it via `openai.STT(base_url=...)`, no plugin changes.

English only, tuned for accent robustness. ~2.5 GB VRAM, sub-second per
utterance on an RTX 2000 Ada.

## Run on the RunPod pod

```bash
# 1. system deps for audio decode
apt-get update && apt-get install -y ffmpeg libsndfile1

# 2. python deps (torch+CUDA come from the pod's base image)
pip install -r requirements.txt

# 3. serve on :9000, all interfaces so the agent worker can reach it
#    first run downloads the model (~2.5 GB) from NGC/HuggingFace
uvicorn server:app --host 0.0.0.0 --port 9000
```

Leave it running (use `tmux`/`nohup` for a detached process). Set
`PARAKEET_MODEL` to override the model id.

## Verify

```bash
# health (model loaded?)
curl -s localhost:9000/health

# transcribe a wav
curl -s -F "file=@sample.wav" -F "model=parakeet" \
  http://localhost:9000/v1/audio/transcriptions
# -> {"text":"..."}
```

## Expose to the agent

The agent worker (frontend VPS) reaches this over the RunPod pod's public
endpoint. Expose TCP port `9000` on the pod (Edit Pod → Expose TCP Ports),
then point the agent at the mapped `host:port` — see `PARAKEET_STT_URL` in
`src/agent_local.py` for the A/B harness, or the `local` STT URL in
`src/agent_aika.py` once validated.

> No auth — the pod's exposed port is public. Stop the pod when not testing,
> or restrict the port to the frontend VPS IP.

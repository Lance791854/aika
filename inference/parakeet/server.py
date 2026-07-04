"""OpenAI-compatible STT wrapper around NVIDIA Parakeet (NeMo).

Exposes POST /v1/audio/transcriptions returning {"text": ...} so the LiveKit
`openai.STT(base_url=...)` plugin can talk to it unchanged — same shape the
agent already uses for Speaches/Whisper. Non-streaming: the agent's VAD has
already segmented each utterance before it's posted here.

Run on the RunPod GPU pod (see README.md). One model, loaded once at startup.
"""

import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parakeet")

MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
TARGET_SR = 16000  # parakeet expects 16 kHz mono

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    # Imported here, not at module top, so the server is importable (and the
    # HTTP/audio path is testable) on a machine without NeMo/CUDA installed.
    import nemo.collections.asr as nemo_asr

    logger.info(f"loading {MODEL_NAME} …")
    _model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    _model.eval()

    # Warm up: the first GPU inference compiles CUDA kernels (~seconds), which
    # otherwise lands on the first real utterance. Run a throwaway transcribe of
    # silence at startup so the first user request is fast.
    try:
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, np.zeros(TARGET_SR // 2, dtype="float32"), TARGET_SR)
            warm_path = tmp.name
        _model.transcribe([warm_path], batch_size=1)
        os.unlink(warm_path)
        logger.info("warmup done")
    except Exception as e:
        logger.warning(f"warmup skipped: {e}")

    logger.info("model ready")
    yield


app = FastAPI(lifespan=lifespan)

# Module-level singletons so the FastAPI param defaults aren't function calls
# in the signature (keeps ruff B008 happy).
_FILE = File(...)
_MODEL_FORM = Form(default=MODEL_NAME)
_LANG_FORM = Form(default="en")
_FORMAT_FORM = Form(default="json")


def _extract_text(result) -> str:
    # NeMo's transcribe() return shape varies by version: a bare string, a
    # Hypothesis with .text, a list, or a (best, all) tuple. Unwrap all of them.
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, (list, tuple)):
        return _extract_text(result[0]) if result else ""
    if isinstance(result, str):
        return result
    return getattr(result, "text", str(result))


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": _model is not None}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = _FILE,
    model: str = _MODEL_FORM,
    language: str = _LANG_FORM,
    response_format: str = _FORMAT_FORM,
):
    raw = await file.read()

    # Decode (LiveKit's openai STT plugin posts a wav container), force mono,
    # resample to 16 kHz.
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # NeMo wants a real file path; delete=False + manual unlink so the handle
    # is closed before transcribe() reads it.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, TARGET_SR)
        path = tmp.name
    try:
        results = _model.transcribe([path], batch_size=1)
    finally:
        os.unlink(path)

    text = _extract_text(results).strip()
    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}

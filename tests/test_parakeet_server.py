"""Local tests for the Parakeet STT wrapper.

These mock the NeMo model, so they validate everything EXCEPT the actual model
inference — HTTP shape, audio decode/mono/resample, response_format, and the
NeMo-return unwrapping — with no GPU or NeMo install. The real model is only
exercised on the RunPod pod.

Skipped unless the server deps are installed:
  uv run --with fastapi --with httpx --with soundfile --with numpy \
    pytest tests/test_parakeet_server.py
"""

import io
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "inference" / "parakeet")
)

import server  # noqa: E402


class _FakeHyp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    """Stands in for the NeMo ASRModel. Records the path it was handed."""

    def __init__(self, text: str = "freezer at minus eighteen") -> None:
        self.text = text
        self.seen_path = None

    def transcribe(self, paths, batch_size=1):
        self.seen_path = paths[0]
        # path must exist at call time (server unlinks it afterwards)
        assert os.path.exists(paths[0]), "temp wav should exist during transcribe"
        return [_FakeHyp(self.text)]


def _wav_bytes(freq=440, sr=16000, seconds=0.5, channels=1) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 2:
        sig = np.stack([sig, sig], axis=1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    # Bypass lifespan (which would import NeMo) by constructing the TestClient
    # WITHOUT the context manager and injecting a fake model directly.
    monkeypatch.setattr(server, "_model", _FakeModel())
    return TestClient(server.app)


def test_transcribe_returns_openai_json_shape(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav_bytes(), "audio/wav")},
        data={"model": "parakeet", "language": "en"},
    )
    assert r.status_code == 200
    assert r.json() == {"text": "freezer at minus eighteen"}


def test_response_format_text(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "text"},
    )
    assert r.status_code == 200
    assert r.text == "freezer at minus eighteen"


def test_temp_file_cleaned_up(client):
    client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav_bytes(), "audio/wav")},
    )
    # server unlinks the temp wav after transcribe
    assert not os.path.exists(server._model.seen_path)


def test_stereo_input_is_accepted(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav_bytes(channels=2), "audio/wav")},
    )
    assert r.status_code == 200


def test_resamples_non_16k_input(client):
    pytest.importorskip("librosa")
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav_bytes(sr=48000), "audio/wav")},
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello", "hello"),
        (_FakeHyp("hi there"), "hi there"),
        ([_FakeHyp("first")], "first"),
        (([_FakeHyp("best")], ["all"]), "best"),
    ],
)
def test_extract_text_unwraps_nemo_shapes(raw, expected):
    assert server._extract_text(raw) == expected

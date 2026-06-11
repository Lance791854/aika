"""flat-file persistence for temperature logs and notes."""

import json
import os
import time
from pathlib import Path

DATA_PATH = Path(os.environ.get("AIKA_DATA_PATH", "data/aika.json"))


def _load() -> dict:
    if not DATA_PATH.exists():
        return {"temperatures": [], "notes": []}
    try:
        return json.loads(DATA_PATH.read_text())
    except json.JSONDecodeError:
        return {"temperatures": [], "notes": []}


def _save(data: dict) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=2))


def add_temperature(location: str, celsius: float) -> None:
    data = _load()
    data["temperatures"].append(
        {"at": time.time(), "location": location.lower(), "celsius": celsius}
    )
    _save(data)


def latest_temperature(location: str) -> dict | None:
    location = location.lower()
    for entry in reversed(_load()["temperatures"]):
        if entry["location"] == location:
            return entry
    return None


def add_note(text: str) -> None:
    data = _load()
    data["notes"].append({"at": time.time(), "text": text})
    _save(data)


def list_notes() -> list[dict]:
    return _load()["notes"]

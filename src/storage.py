"""flat-file persistence for temperature logs and notes."""

import json
import os
import time
from pathlib import Path

DATA_PATH = Path(os.environ.get("AIKA_DATA_PATH", "data/aika.json"))

# safe temperature ranges per FSANZ (Food Standards Australia New Zealand).
# https://www.foodstandards.gov.au/consumer/safety/temperature
#   frozen food kept at -18C or colder
#   refrigerated food at 5C or colder
#   hot food held above 60C
#   cooked poultry to minimum 75C internal
TEMP_RANGES = {
    "freezer": (-25, -18),
    "fridge": (0, 5),
    "cool room": (0, 5),
    "hot hold": (60, 90),
    "chicken": (75, 100),
    "poultry": (75, 100),
    "beef": (63, 100),
    "pork": (63, 100),
    "lamb": (63, 100),
    "fish": (63, 100),
}


def check_range(location: str, celsius: float) -> str | None:
    """Return a warning string if outside the safe range, otherwise None."""
    rng = TEMP_RANGES.get(location.lower())
    if rng is None:
        return None
    low, high = rng
    if celsius < low:
        return f"{location} is below safe range, should be at least {low} degrees"
    if celsius > high:
        return f"{location} is above safe range, should be at most {high} degrees"
    return None


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


def recent_temperatures(limit: int = 5) -> list[dict]:
    """latest reading per location, newest first."""
    seen: set[str] = set()
    result: list[dict] = []
    for entry in reversed(_load()["temperatures"]):
        if entry["location"] in seen:
            continue
        seen.add(entry["location"])
        result.append(entry)
        if len(result) >= limit:
            break
    return result


def recent_notes(limit: int = 5) -> list[dict]:
    return list(reversed(_load()["notes"]))[:limit]

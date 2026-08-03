"""flat-file persistence for temperature logs and notes."""

import json
import os
import re
import time
from pathlib import Path

DATA_PATH = Path(os.environ.get("AIKA_DATA_PATH", "data/aika.json"))
_BASE_PATH = DATA_PATH


def scope_name(chef: str) -> str:
    """A chef name cleaned down to something safe for a filename."""
    return re.sub(r"[^a-z0-9]+", "-", chef.lower()).strip("-")[:24].strip("-")


def set_scope(chef: str) -> None:
    """Point storage at one chef's file. Empty name = the shared default.

    Everything is per chef for now. If some things should be shared later
    (freezer temps, stock), route just those functions at a kitchen file.
    """
    global DATA_PATH
    n = scope_name(chef)
    DATA_PATH = _BASE_PATH.with_name(f"aika-chef-{n}.json") if n else _BASE_PATH


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


# seconds before an uncorrected bad reading gets a repeat spoken warning.
ALERT_COOLDOWN = 300


def overdue_temperatures(
    last_alerted: dict[str, float],
    now: float | None = None,
    cooldown: float = ALERT_COOLDOWN,
) -> list[dict]:
    """Locations whose latest reading is still out of range and due a re-alert.

    Due means neither the reading itself nor the last alert for that location
    is fresher than `cooldown` — a corrective reading clears it, a fresh alert
    postpones it.
    """
    now = time.time() if now is None else now
    seen: set[str] = set()
    due: list[dict] = []
    for entry in reversed(_load()["temperatures"]):
        loc = entry["location"]
        if loc in seen:
            continue
        seen.add(loc)
        warning = check_range(loc, entry["celsius"])
        if not warning:
            continue
        if now - max(entry["at"], last_alerted.get(loc, 0)) < cooldown:
            continue
        due.append({"location": loc, "celsius": entry["celsius"], "warning": warning})
    return due


def _load() -> dict:
    if not DATA_PATH.exists():
        return {"temperatures": [], "notes": [], "unhandled": []}
    try:
        return json.loads(DATA_PATH.read_text())
    except json.JSONDecodeError:
        return {"temperatures": [], "notes": [], "unhandled": []}


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


def add_stock(item: str, quantity: str = "", urgency: str = "normal") -> None:
    data = _load()
    data.setdefault("stock", []).append(
        {
            "at": time.time(),
            "item": item,
            "quantity": quantity,
            "urgency": urgency.lower(),
        }
    )
    _save(data)


def recent_stock(limit: int = 10) -> list[dict]:
    return list(reversed(_load().get("stock", [])))[:limit]


def delete_stock(at: float) -> None:
    data = _load()
    data["stock"] = [e for e in data.get("stock", []) if e.get("at") != at]
    _save(data)


def clear_stock() -> None:
    data = _load()
    data["stock"] = []
    _save(data)


def last_alerted() -> dict[str, float]:
    """Per-location timestamps of the last spoken re-alert."""
    return _load().get("alerts", {})


def mark_alerted(location: str, at: float | None = None) -> None:
    data = _load()
    data.setdefault("alerts", {})[location.lower()] = time.time() if at is None else at
    _save(data)


def shift_summary() -> str:
    """Spoken end-of-shift report: temps (flagging out-of-range), notes, unhandled."""
    parts = []
    temps = recent_temperatures(10)
    if temps:
        bits = []
        warnings = []
        for t in temps:
            c = t["celsius"]
            c = int(c) if c == int(c) else c
            bits.append(f"{t['location']} at {c}")
            w = check_range(t["location"], t["celsius"])
            if w:
                warnings.append(w)
        s = "Temperatures. " + ", ".join(bits) + "."
        if warnings:
            s += " Warning. " + ". ".join(warnings) + "."
        else:
            s += " All in range."
        parts.append(s)
    else:
        parts.append("No temperatures logged.")
    notes = recent_notes(10)
    if notes:
        parts.append("Notes. " + ". ".join(n["text"] for n in reversed(notes)) + ".")
    else:
        parts.append("No notes.")
    stock = recent_stock(10)
    if stock:
        bits = []
        for e in reversed(stock):
            b = e["item"]
            if e["quantity"]:
                b += f", {e['quantity']}"
            if e["urgency"] == "urgent":
                b += ", urgent"
            bits.append(b)
        parts.append("Stock needed. " + ". ".join(bits) + ".")
    unhandled = recent_unhandled(10)
    if unhandled:
        n = len(unhandled)
        s = "s" if n != 1 else ""
        parts.append(f"{n} request{s} I couldn't handle — check the panel.")
    return " ".join(parts)


def add_unhandled(text: str) -> None:
    """Record a request AIKA couldn't fulfil, for reviewing coverage gaps."""
    data = _load()
    data.setdefault("unhandled", []).append({"at": time.time(), "text": text})
    _save(data)


def recent_unhandled(limit: int = 5) -> list[dict]:
    return list(reversed(_load().get("unhandled", [])))[:limit]


def clear_unhandled() -> None:
    data = _load()
    data["unhandled"] = []
    _save(data)


def delete_unhandled(at: float) -> None:
    data = _load()
    data["unhandled"] = [u for u in data.get("unhandled", []) if u.get("at") != at]
    _save(data)


def delete_note(at: float) -> None:
    data = _load()
    data["notes"] = [n for n in data.get("notes", []) if n.get("at") != at]
    _save(data)


def clear_notes() -> None:
    data = _load()
    data["notes"] = []
    _save(data)


def delete_temperature_location(location: str) -> None:
    """Remove every reading for a location — clears its card entirely."""
    location = location.lower()
    data = _load()
    data["temperatures"] = [
        t for t in data.get("temperatures", []) if t.get("location") != location
    ]
    _save(data)


def clear_temperatures() -> None:
    data = _load()
    data["temperatures"] = []
    _save(data)

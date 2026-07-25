"""Unattended temperature re-alerting.

storage.overdue_temperatures decides which locations deserve a repeat spoken
warning: latest reading out of range, and neither the reading itself nor the
last alert is fresher than the cooldown. Pure logic — the agent's background
loop just speaks whatever this returns.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import storage

COOLDOWN = 60


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Point storage at a throwaway file so tests never touch data/aika.json."""
    monkeypatch.setattr(storage, "DATA_PATH", tmp_path / "aika.json")


def _at(location: str) -> float:
    return storage.latest_temperature(location)["at"]


def test_bad_reading_due_after_cooldown() -> None:
    storage.add_temperature("freezer", -12)
    due = storage.overdue_temperatures(
        {}, now=_at("freezer") + COOLDOWN + 1, cooldown=COOLDOWN
    )
    assert [d["location"] for d in due] == ["freezer"]
    assert "above safe range" in due[0]["warning"]


def test_fresh_bad_reading_not_due_yet() -> None:
    # log_temperature already spoke a warning at log time — don't nag straight away.
    storage.add_temperature("freezer", -12)
    assert (
        storage.overdue_temperatures({}, now=_at("freezer") + 5, cooldown=COOLDOWN)
        == []
    )


def test_in_range_reading_never_due() -> None:
    storage.add_temperature("fridge", 4)
    assert (
        storage.overdue_temperatures(
            {}, now=_at("fridge") + COOLDOWN * 10, cooldown=COOLDOWN
        )
        == []
    )


def test_corrected_reading_clears_the_alert() -> None:
    storage.add_temperature("freezer", -12)
    storage.add_temperature("freezer", -20)
    assert (
        storage.overdue_temperatures(
            {}, now=_at("freezer") + COOLDOWN * 10, cooldown=COOLDOWN
        )
        == []
    )


def test_recent_alert_suppresses_until_cooldown_then_realerts() -> None:
    storage.add_temperature("freezer", -12)
    at = _at("freezer")
    alerted = {"freezer": at + COOLDOWN + 1}
    assert (
        storage.overdue_temperatures(alerted, now=at + COOLDOWN + 2, cooldown=COOLDOWN)
        == []
    )
    due = storage.overdue_temperatures(
        alerted, now=at + COOLDOWN * 2 + 2, cooldown=COOLDOWN
    )
    assert [d["location"] for d in due] == ["freezer"]


def test_unknown_location_never_due() -> None:
    storage.add_temperature("spice rack", 40)
    assert (
        storage.overdue_temperatures(
            {}, now=_at("spice rack") + COOLDOWN * 10, cooldown=COOLDOWN
        )
        == []
    )


def test_multiple_bad_locations_all_reported() -> None:
    storage.add_temperature("freezer", -12)
    storage.add_temperature("fridge", 9)
    now = _at("fridge") + COOLDOWN + 1
    locs = {
        d["location"]
        for d in storage.overdue_temperatures({}, now=now, cooldown=COOLDOWN)
    }
    assert locs == {"freezer", "fridge"}

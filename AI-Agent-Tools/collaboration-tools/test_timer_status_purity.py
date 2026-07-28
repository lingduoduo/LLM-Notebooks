"""Regression test: get_timer_status must not mutate stored timer state.

`remaining_seconds` is derived state that is only meaningful at read time.
Writing it into the stored record leaked a stale countdown into list_timers()
and into the persisted timers.json, where it kept whatever value happened to be
current the last time somebody called get_timer_status().
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import timer_tools as t  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch, tmp_path):
    from config import config

    monkeypatch.setattr(config.timer, "storage_path", str(tmp_path / "timers.json"))
    t._active_timers.clear()
    t._timer_tasks.clear()
    yield
    for task in t._timer_tasks.values():
        task.cancel()
    t._active_timers.clear()
    t._timer_tasks.clear()


def test_get_timer_status_does_not_write_into_stored_record():
    async def scenario():
        created = await t.set_timer(duration_seconds=60, timer_name="probe")
        timer_id = created["timer_id"]

        status = await t.get_timer_status(timer_id)
        return timer_id, status

    timer_id, status = asyncio.run(scenario())

    assert "remaining_seconds" in status["timer"], "caller should still see the countdown"
    assert "remaining_seconds" not in t._active_timers[timer_id], (
        "derived countdown leaked into the stored record"
    )


def test_listed_timers_carry_no_stale_countdown():
    async def scenario():
        created = await t.set_timer(duration_seconds=60, timer_name="probe")
        await t.get_timer_status(created["timer_id"])
        return await t.list_timers()

    listing = asyncio.run(scenario())
    for timer in listing["timers"]:
        assert "remaining_seconds" not in timer


def test_status_result_is_a_copy_not_the_live_record():
    async def scenario():
        created = await t.set_timer(duration_seconds=60, timer_name="probe")
        timer_id = created["timer_id"]
        status = await t.get_timer_status(timer_id)
        status["timer"]["name"] = "mutated by caller"
        return timer_id

    timer_id = asyncio.run(scenario())
    assert t._active_timers[timer_id]["name"] == "probe"

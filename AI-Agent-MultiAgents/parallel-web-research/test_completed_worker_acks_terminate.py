"""A worker must answer a terminate that lands after it finished on its own.

The coordinator's expected-ack set is a snapshot taken when the winner settles
(see test_completed_worker_ack_regression.py), so a worker that was live at that
moment is expected to answer even if it then completed by itself. The worker
used to acknowledge only from its cancellation path, so a worker that published
``not_found`` and then received the terminate went straight to cleanup and never
answered -- failing the `cascade_loser_acknowledgements` acceptance gate on a run
that was actually correct.
"""

import asyncio

import pytest

import agents
from agents import TaskState, WorkerAgent
from message_bus import BROADCAST, Envelope, MessageBus
from sources import Website


class _StubPage:
    url = "https://site.edu/final"

    async def goto(self, *_args, **_kwargs):
        return type("Response", (), {"status": 200})()

    def locator(self, _selector):
        async def inner_text(timeout=None):
            return "rendered page text"
        return type("Locator", (), {"inner_text": staticmethod(inner_text)})()


class _StubContext:
    def __init__(self, pool):
        self._pool = pool

    async def new_page(self):
        return _StubPage()

    async def close(self):
        self._pool.closed += 1


class _StubBrowserPool:
    def __init__(self):
        self.closed = 0

    async def new_context(self):
        return _StubContext(self)

    async def mark_closed(self):
        return None


def _worker(bus):
    return WorkerAgent("worker-1", Website("s1", "https://site.edu", "College 1"),
                       bus, "target", _StubBrowserPool(), timeout=5)


async def _run_finding_nothing(monkeypatch, terminate_on_success):
    """Run a worker to a clean not_found finish.

    With ``terminate_on_success`` the terminate is published from inside the
    SUCCEEDED status report -- after ``not_found`` has gone out and before the
    cleanup block -- which is precisely the window the worker used to ignore.
    """
    bus = MessageBus(verbose=False)
    worker = _worker(bus)

    async def fake_extract(*_args, **_kwargs):
        return {"found": False, "reason": "not on this page"}

    monkeypatch.setattr(agents, "extract_profile", fake_extract)

    original_report = worker.report

    async def report(state, note):
        await original_report(state, note)
        if terminate_on_success and state is TaskState.SUCCEEDED:
            await bus.send("coordinator", BROADCAST, "terminate",
                           {"reason": "target_found_by_worker-2", "winner": "worker-2"})

    monkeypatch.setattr(worker, "report", report)

    await bus.send("coordinator", "worker-1", "task_assigned",
                   {"target": "target", "url": worker.site.url, "task_id": "worker-1"})
    await asyncio.wait_for(worker.run(), timeout=5)
    return bus, worker


@pytest.mark.asyncio
async def test_worker_acknowledges_a_terminate_delivered_after_it_finished(monkeypatch):
    bus, worker = await _run_finding_nothing(monkeypatch, terminate_on_success=True)

    # It reported its own result -- it was never cancelled.
    assert any(message.type == "not_found" for message in bus.history)
    assert worker.terminate.is_set()

    acks = [message for message in bus.history
            if message.type == "ack" and message.sender_id == "worker-1"]
    assert len(acks) == 1, "a delivered terminate must be acknowledged exactly once"
    assert acks[0].payload["acked"] == "terminate"

    # And it still released its browser context.
    closed = [message for message in bus.history if message.type == "resource_closed"]
    assert closed and closed[0].payload["browser_context_closed"] is True


@pytest.mark.asyncio
async def test_worker_does_not_acknowledge_when_no_terminate_arrived(monkeypatch):
    """The acknowledgement must mean something: no terminate, no ack."""
    bus, worker = await _run_finding_nothing(monkeypatch, terminate_on_success=False)

    assert not worker.terminate.is_set()
    assert not [message for message in bus.history if message.type == "ack"]
    assert any(message.type == "not_found" for message in bus.history)


@pytest.mark.asyncio
async def test_unread_terminate_left_in_the_inbox_is_still_noticed():
    """The signal task can be cancelled before it dequeues the terminate."""
    bus = MessageBus(verbose=False)
    worker = _worker(bus)
    worker.sub.inbox.put_nowait(Envelope(
        sender_id="coordinator", target=BROADCAST, type="terminate",
        payload={"reason": "target_found_by_worker-2"},
    ))

    worker._absorb_pending_terminate()

    assert worker.terminate.is_set()
    assert worker._termination_reason == "target_found_by_worker-2"


@pytest.mark.asyncio
async def test_acknowledgement_is_idempotent():
    """Both the cancellation path and cleanup can reach it; only one ack may go out."""
    bus = MessageBus(verbose=False)
    worker = _worker(bus)

    await worker._acknowledge_terminate()
    await worker._acknowledge_terminate()

    assert len([m for m in bus.history if m.type == "ack"]) == 1

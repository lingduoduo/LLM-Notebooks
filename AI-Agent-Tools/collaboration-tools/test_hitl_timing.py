"""Regression tests for HITL response timing and request lifecycle.

`_wait_for_admin_response` used to poll on a hardcoded `asyncio.sleep(2)`. The
first check ran before any answer could exist, and the sleep then overshot any
deadline of 2s or less, so a response that genuinely arrived in time came back
as a timeout:

    timeout=2s, admin approved at 1.0s -> approved=False, timeout=True

Because HITL fails closed, that silently converted real approvals into denials.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import hitl_tools as h  # noqa: E402


@pytest.fixture(autouse=True)
def offline_and_clean():
    """Silence admin notifications and isolate the module-level registries."""
    from config import config

    config.hitl.admin_email = None
    config.hitl.webhook_url = None
    config.im.telegram_bot_token = None
    config.im.slack_webhook_url = None
    config.email.sendgrid_api_key = None
    config.email.smtp_username = None
    config.email.smtp_password = None

    # getattr: _response_events is part of the fix, so these tests must still be
    # loadable against the pre-fix module in order to demonstrate the failure.
    def _reset():
        h._pending_requests.clear()
        events = getattr(h, "_response_events", None)
        if events is not None:
            events.clear()

    _reset()
    yield
    _reset()


async def _respond_after(delay, approve=True, notes="answered"):
    await asyncio.sleep(delay)
    pending = await h.list_pending_requests()
    for req in pending["requests"]:
        await h.respond_to_request(req["request_id"], approve, notes)


class TestInTimeResponsesAreHonored:
    @pytest.mark.parametrize("timeout_seconds", [1, 2, 3, 5])
    def test_approval_before_deadline_is_not_reported_as_timeout(self, timeout_seconds):
        """The exact bug: timeout<=2s used to lose an approval that arrived at 0.5s."""

        async def scenario():
            return await asyncio.gather(
                h.request_admin_approval("ship it?", timeout_seconds=timeout_seconds),
                _respond_after(0.5, approve=True),
            )

        result, _ = asyncio.run(scenario())
        assert result["approved"] is True, f"approval lost at timeout={timeout_seconds}s"
        assert not result.get("timeout")
        assert result["admin_notes"] == "answered"

    def test_rejection_before_deadline_is_honored(self):
        async def scenario():
            return await asyncio.gather(
                h.request_admin_approval("drop the table?", timeout_seconds=2),
                _respond_after(0.5, approve=False, notes="absolutely not"),
            )

        result, _ = asyncio.run(scenario())
        assert result["approved"] is False
        assert not result.get("timeout")
        assert result["reason"] == "absolutely not"

    def test_waiter_wakes_promptly_rather_than_polling(self):
        """An answer at 0.1s should return in well under the old 2s poll interval."""

        async def scenario():
            loop = asyncio.get_event_loop()
            start = loop.time()
            result, _ = await asyncio.gather(
                h.request_admin_approval("quick?", timeout_seconds=30),
                _respond_after(0.1, approve=True),
            )
            return result, loop.time() - start

        result, elapsed = asyncio.run(scenario())
        assert result["approved"] is True
        assert elapsed < 1.0, f"took {elapsed:.2f}s; waiter is still polling slowly"


class TestGenuineTimeouts:
    def test_no_response_still_times_out_and_fails_closed(self):
        result = asyncio.run(h.request_admin_approval("silence?", timeout_seconds=1))
        assert result["approved"] is False
        assert result["timeout"] is True

    def test_zero_timeout_returns_immediately_instead_of_waiting_an_hour(self):
        """`or` used to swallow an explicit 0 and substitute the 3600s default."""

        async def scenario():
            loop = asyncio.get_event_loop()
            start = loop.time()
            result = await h.request_admin_approval("no waiting", timeout_seconds=0)
            return result, loop.time() - start

        result, elapsed = asyncio.run(scenario())
        assert result["approved"] is False
        assert result["timeout"] is True
        assert elapsed < 1.0

    def test_none_timeout_uses_configured_default(self):
        from config import config

        config.hitl.timeout_seconds = 1
        result = asyncio.run(h.request_admin_approval("default", timeout_seconds=None))
        assert result["timeout"] is True


class TestRequestLifecycle:
    def test_settled_request_cannot_be_answered_twice(self):
        async def scenario():
            result, _ = await asyncio.gather(
                h.request_admin_approval("once", timeout_seconds=5),
                _respond_after(0.1, approve=True),
            )
            rid = result["request_id"]
            return await h.respond_to_request(rid, False, "changed my mind")

        second = asyncio.run(scenario())
        assert second["success"] is False
        assert "already approved" in second["message"]

    def test_resolved_requests_are_pruned(self):
        async def scenario():
            for _ in range(h._MAX_RESOLVED_REQUESTS + 10):
                await h.request_admin_approval("noise", timeout_seconds=0)

        asyncio.run(scenario())
        assert len(h._pending_requests) <= h._MAX_RESOLVED_REQUESTS

    def test_admin_input_receives_the_answer(self):
        async def scenario():
            return await asyncio.gather(
                h.request_admin_input("what region?", timeout_seconds=5),
                _respond_after(0.1, approve=True, notes="eu-west-1"),
            )

        result, _ = asyncio.run(scenario())
        assert result["success"] is True
        assert result["input"] == "eu-west-1"

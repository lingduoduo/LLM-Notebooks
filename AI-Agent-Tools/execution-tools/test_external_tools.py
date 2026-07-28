"""Tests for the external integration tools.

These tests never touch the network. Credentials are neutralized by the
autouse fixture in conftest.py, and the cases that need a client install a
fake one.

The previous versions of the validation tests passed for the wrong reason:
they asserted `not result["success"]` for an invalid datetime, but the call
had already failed at "Google Calendar libraries not installed" long before
any datetime was parsed. They would have kept passing if datetime validation
were deleted outright.
"""

from datetime import datetime
from pathlib import Path

import pytest

from config import Config
from external_tools import ExternalTools
from llm_helper import LLMHelper


@pytest.fixture
def tools():
    return ExternalTools(LLMHelper())


@pytest.fixture
def no_approval(monkeypatch):
    monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)


class FakeEvents:
    def __init__(self, created):
        self.created = created
        self.calls = []

    def insert(self, calendarId, body):
        self.calls.append({"calendarId": calendarId, "body": body})
        return type("Request", (), {"execute": lambda _self: self.created})()


class FakeCalendarService:
    def __init__(self, created=None):
        self._events = FakeEvents(
            created or {"id": "evt-1", "htmlLink": "https://calendar.example/evt-1"}
        )

    def events(self):
        return self._events


class TestSuiteIsHermetic:
    """Credentials from .env must never reach a test.

    PyGithub is not currently installed, so these calls bail early by
    accident. Installing requirements.txt would remove that accident and the
    external-tool tests would start making live authenticated requests.
    """

    def test_credentials_are_neutralized(self):
        assert Config.OPENAI_API_KEY is None
        assert Config.GITHUB_TOKEN is None
        assert not Path(Config.GOOGLE_CALENDAR_CREDENTIALS_FILE).exists()

    def test_environment_variables_are_cleared(self):
        import os

        assert os.getenv("OPENAI_API_KEY") is None
        assert os.getenv("GITHUB_TOKEN") is None


class TestCalendarInputValidation:
    """Input validation must run before any client is built.

    Validating after client setup means the checks are unreachable for anyone
    without credentials -- including every offline test run.
    """

    async def test_unparseable_start_time_is_rejected(self, tools):
        result = await tools.google_calendar_add(
            summary="Test", start_time="invalid-datetime", end_time="2026-10-01T11:00:00"
        )

        assert result["success"] is False
        assert "datetime" in result["error"].lower()
        assert "invalid-datetime" in result["error"]

    async def test_end_before_start_is_rejected(self, tools):
        result = await tools.google_calendar_add(
            summary="Test",
            start_time="2026-10-01T11:00:00",
            end_time="2026-10-01T10:00:00",
        )

        assert result["success"] is False
        assert "after start time" in result["error"]

    async def test_equal_start_and_end_is_rejected(self, tools):
        result = await tools.google_calendar_add(
            summary="Test",
            start_time="2026-10-01T10:00:00",
            end_time="2026-10-01T10:00:00",
        )

        assert result["success"] is False
        assert "after start time" in result["error"]

    async def test_missing_credentials_are_reported_for_valid_input(self, tools, no_approval):
        """Valid input with no credentials must fail on credentials, not parsing."""
        result = await tools.google_calendar_add(
            summary="Test",
            start_time="2026-10-01T10:00:00",
            end_time="2026-10-01T11:00:00",
        )

        assert result["success"] is False
        assert "Google Calendar" in result["error"]

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2026-10-01T10:00:00", datetime(2026, 10, 1, 10, 0, 0)),
            ("2026-10-01 10:00:00", datetime(2026, 10, 1, 10, 0, 0)),
            ("2026-10-01 10:00", datetime(2026, 10, 1, 10, 0)),
            ("2026-10-01", datetime(2026, 10, 1, 0, 0)),
        ],
    )
    def test_accepted_datetime_formats(self, tools, value, expected):
        assert tools._parse_datetime(value) == expected

    def test_unknown_datetime_format_raises(self, tools):
        with pytest.raises(ValueError, match="Could not parse datetime"):
            tools._parse_datetime("next tuesday")


class TestCalendarEventCreation:
    async def test_event_is_created_from_the_given_fields(self, tools, no_approval, monkeypatch):
        service = FakeCalendarService()
        monkeypatch.setattr(tools, "_get_google_calendar_service", lambda: service)

        result = await tools.google_calendar_add(
            summary="Design review",
            start_time="2026-10-01T10:00:00",
            end_time="2026-10-01T11:00:00",
            description="Quarterly",
            location="Room 4",
        )

        assert result["success"] is True
        assert result["event_id"] == "evt-1"

        body = service.events().calls[0]["body"]
        assert body["summary"] == "Design review"
        assert body["description"] == "Quarterly"
        assert body["location"] == "Room 4"
        assert body["start"]["dateTime"] == "2026-10-01T10:00:00"

    async def test_optional_fields_are_omitted_when_absent(self, tools, no_approval, monkeypatch):
        service = FakeCalendarService()
        monkeypatch.setattr(tools, "_get_google_calendar_service", lambda: service)

        await tools.google_calendar_add(
            summary="Standup",
            start_time="2026-10-01T10:00:00",
            end_time="2026-10-01T10:15:00",
        )

        body = service.events().calls[0]["body"]
        assert "description" not in body
        assert "location" not in body

    async def test_creation_requires_approval(self, tools, monkeypatch):
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)
        monkeypatch.setattr(
            LLMHelper, "request_approval", lambda self, op, details: (False, "denied by test")
        )
        service = FakeCalendarService()
        monkeypatch.setattr(tools, "_get_google_calendar_service", lambda: service)

        result = await tools.google_calendar_add(
            summary="Test",
            start_time="2026-10-01T10:00:00",
            end_time="2026-10-01T11:00:00",
        )

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert service.events().calls == []


class TestGitHubInputValidation:
    async def test_malformed_repository_name_is_rejected(self, tools):
        result = await tools.github_create_pr(
            repo_name="not-a-repo-path",
            title="Test",
            body="Body",
            head_branch="feature",
        )

        assert result["success"] is False
        assert "owner/repo" in result["error"]

    async def test_missing_token_is_reported_for_valid_input(self, tools, no_approval):
        result = await tools.github_create_pr(
            repo_name="owner/repo",
            title="Test",
            body="Body",
            head_branch="feature",
        )

        assert result["success"] is False
        assert "GitHub" in result["error"]

    async def test_pr_creation_requires_approval(self, tools, monkeypatch):
        """No live call may happen before the gate; the client is never built."""
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)
        monkeypatch.setattr(
            LLMHelper, "request_approval", lambda self, op, details: (False, "denied by test")
        )

        def explode():
            raise AssertionError("no GitHub client may be built for a refused PR")

        monkeypatch.setattr(tools, "_get_github_client", explode)

        result = await tools.github_create_pr(
            repo_name="owner/repo",
            title="Test",
            body="Body",
            head_branch="feature",
        )

        assert result["success"] is False
        assert "not approved" in result["error"]

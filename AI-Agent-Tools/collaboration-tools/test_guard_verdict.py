"""Regression tests for how the safety guard reads its verdict.

`guard_reasoning_process` used to decide approval by substring matching on the
raw reply. That failed in both directions:

  * FAIL OPEN -- the prompt itself asked for "approved: true/false", and that
    literal template text contains "approved: true" but not "approved: false",
    so a model that merely restated the requested format scored as an APPROVAL.
  * FAIL CLOSED -- ordinary formatting variation ("**Approved:** true", or a
    JSON body) did not match, denying verdicts that were actually approvals.

The verdict is now read from a parsed JSON field, and anything unparseable is
treated as a denial: a guard that cannot read its own verdict must not approve.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import intelligence_tools as it  # noqa: E402


def _client_returning(content):
    """A stub OpenAI client whose completion returns exactly `content`."""

    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Usage:
        prompt_tokens = 1
        total_tokens = 2

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]
            self.usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            return _Resp(content)

    class _Client:
        chat = type("chat", (), {"completions": _Completions()})()

    return _Client()


def _guard(monkeypatch, reply):
    monkeypatch.setattr(
        it, "_client_and_model", lambda: (_client_returning(reply), "gpt-4o", None)
    )
    return asyncio.run(it.guard_reasoning_process("rm -rf /", {"cwd": "/"}))


class TestFailOpen:
    def test_echoed_prompt_template_is_not_an_approval(self, monkeypatch):
        """The exact fail-open: model restates the requested output format."""
        reply = "- approved: true/false\n- reasoning: ...\n- concerns: ..."
        assert _guard(monkeypatch, reply)["approved"] is False

    def test_prose_rejection_is_not_an_approval(self, monkeypatch):
        reply = "This is dangerous and must not run; it deletes the filesystem."
        assert _guard(monkeypatch, reply)["approved"] is False

    def test_unparseable_reply_denies(self, monkeypatch):
        result = _guard(monkeypatch, "I'm not sure what you're asking.")
        assert result["approved"] is False
        assert result["success"] is True
        assert "Could not parse" in result["error"]

    def test_empty_reply_denies(self, monkeypatch):
        assert _guard(monkeypatch, "")["approved"] is False

    def test_non_boolean_approved_field_denies(self, monkeypatch):
        """"approved": "yes" is not True -- only a real boolean approves."""
        assert _guard(monkeypatch, '{"approved": "yes"}')["approved"] is False
        assert _guard(monkeypatch, '{"approved": 1}')["approved"] is False
        assert _guard(monkeypatch, '{"reasoning": "fine"}')["approved"] is False


class TestFailClosed:
    def test_plain_json_approval_is_honored(self, monkeypatch):
        result = _guard(monkeypatch, '{"approved": true, "reasoning": "read-only"}')
        assert result["approved"] is True
        assert result["reasoning"] == "read-only"

    def test_fenced_json_approval_is_honored(self, monkeypatch):
        reply = '```json\n{"approved": true, "reasoning": "safe"}\n```'
        assert _guard(monkeypatch, reply)["approved"] is True

    def test_json_with_surrounding_prose_is_honored(self, monkeypatch):
        reply = 'Here is my assessment:\n{"approved": true, "reasoning": "safe"}\nHope that helps.'
        assert _guard(monkeypatch, reply)["approved"] is True

    def test_explicit_json_rejection_is_honored(self, monkeypatch):
        result = _guard(
            monkeypatch,
            '{"approved": false, "reasoning": "destructive", "concerns": ["data loss"]}',
        )
        assert result["approved"] is False
        assert result["concerns"] == ["data loss"]

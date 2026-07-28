"""Tests for the file system tools.

Regression coverage for the overwrite bug: ``write_file(overwrite=False)``
only *requested approval* before clobbering an existing file, and had no
branch that actually refused. With approval disabled -- the configuration the
README recommends for offline demos -- passing ``overwrite=False`` silently
destroyed the existing file and reported ``success: True``.
"""

import pytest

from config import Config
from file_tools import FileTools
from llm_helper import LLMHelper


@pytest.fixture
def tools(workspace):
    return FileTools(LLMHelper())


class RecordedApprovals(list):
    """Approval requests, plus the verdict the stub should return."""

    verdict = {"approved": True, "reason": "approved by test"}


@pytest.fixture
def approvals(monkeypatch):
    """Capture approval requests; the verdict is set per test."""
    requests = RecordedApprovals()
    requests.verdict = {"approved": True, "reason": "approved by test"}

    def fake_request(self, operation, details):
        requests.append({"operation": operation, "details": details})
        return requests.verdict["approved"], requests.verdict["reason"]

    monkeypatch.setattr(LLMHelper, "request_approval", fake_request)
    monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)
    return requests


class TestWriteFile:
    async def test_writes_a_new_file(self, tools, workspace):
        result = await tools.write_file(path="notes.txt", content="hello")

        assert result["success"] is True
        assert (workspace / "notes.txt").read_text() == "hello"

    async def test_overwrite_false_refuses_to_clobber(self, tools, workspace, monkeypatch):
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)
        target = workspace / "notes.txt"
        target.write_text("ORIGINAL")

        result = await tools.write_file(path="notes.txt", content="CLOBBERED")

        assert result["success"] is False
        assert "exists" in result["error"]
        assert target.read_text() == "ORIGINAL"

    async def test_overwrite_false_refuses_even_when_approval_is_enabled(
        self, tools, workspace, approvals
    ):
        """An explicit overwrite=False is the caller's decision, not the LLM's."""
        target = workspace / "notes.txt"
        target.write_text("ORIGINAL")

        result = await tools.write_file(path="notes.txt", content="CLOBBERED")

        assert result["success"] is False
        assert target.read_text() == "ORIGINAL"
        assert approvals == []

    async def test_overwrite_true_requests_approval_and_proceeds(
        self, tools, workspace, approvals
    ):
        target = workspace / "notes.txt"
        target.write_text("ORIGINAL")

        result = await tools.write_file(
            path="notes.txt", content="REPLACED", overwrite=True
        )

        assert result["success"] is True
        assert target.read_text() == "REPLACED"
        assert approvals[0]["operation"] == "file_overwrite"

    async def test_overwrite_true_respects_a_denied_approval(
        self, tools, workspace, approvals
    ):
        approvals.verdict["approved"] = False
        approvals.verdict["reason"] = "too risky"
        target = workspace / "notes.txt"
        target.write_text("ORIGINAL")

        result = await tools.write_file(
            path="notes.txt", content="REPLACED", overwrite=True
        )

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert target.read_text() == "ORIGINAL"

    async def test_rejects_paths_outside_the_workspace(self, tools):
        result = await tools.write_file(path="/tmp/outside_workspace.txt", content="x")

        assert result["success"] is False
        assert "outside workspace" in result["error"]

    async def test_rejects_traversal_out_of_the_workspace(self, tools):
        result = await tools.write_file(path="../escaped.txt", content="x")

        assert result["success"] is False
        assert "outside workspace" in result["error"]

    async def test_python_syntax_error_blocks_the_write(self, tools, workspace):
        result = await tools.write_file(path="broken.py", content="def broken(:\n")

        assert result["success"] is False
        assert result["verification"] == "failed"
        assert not (workspace / "broken.py").exists()

    async def test_valid_python_is_written_and_marked_passed(self, tools, workspace):
        result = await tools.write_file(path="ok.py", content="x = 1\n")

        assert result["success"] is True
        assert result["verification"] == "passed"
        assert (workspace / "ok.py").read_text() == "x = 1\n"

    async def test_unverifiable_javascript_is_written_and_labelled(self, tools, workspace):
        """No LLM available: write it, but do not claim it was verified."""
        result = await tools.write_file(path="app.js", content="const x = 1;\n")

        assert result["success"] is True
        assert result["verification"] == "unverified"
        assert (workspace / "app.js").exists()


class TestEditFile:
    async def test_replaces_matching_text(self, tools, workspace):
        target = workspace / "note.txt"
        target.write_text("hello world\n")

        result = await tools.edit_file(path="note.txt", search="hello", replace="hi")

        assert result["success"] is True
        assert target.read_text() == "hi world\n"
        assert "diff_preview" in result

    async def test_missing_search_text_is_reported(self, tools, workspace):
        target = workspace / "note.txt"
        target.write_text("hello world\n")

        result = await tools.edit_file(path="note.txt", search="absent", replace="x")

        assert result["success"] is False
        assert "not found" in result["error"]
        assert target.read_text() == "hello world\n"

    async def test_edit_that_breaks_python_syntax_is_rejected(self, tools, workspace):
        target = workspace / "mod.py"
        target.write_text("value = 1\n")

        result = await tools.edit_file(path="mod.py", search="value = 1", replace="def (:")

        assert result["success"] is False
        assert target.read_text() == "value = 1\n"

    async def test_missing_file_is_reported(self, tools):
        result = await tools.edit_file(path="absent.txt", search="a", replace="b")

        assert result["success"] is False
        assert "does not exist" in result["error"]

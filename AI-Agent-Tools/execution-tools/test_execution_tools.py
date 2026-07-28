"""Tests for the generic execution tools (code interpreter, virtual terminal)."""

import pytest

from config import Config
from execution_tools import ExecutionTools
from llm_helper import LLMHelper


@pytest.fixture
def tools(offline_safety):
    return ExecutionTools(LLMHelper())


@pytest.fixture
def analysis_stub(monkeypatch):
    """Record analyze_error calls without contacting a provider."""
    calls = []

    def fake_analyze(self, tool_name, command, error_output):
        calls.append({"tool": tool_name, "command": command, "error": error_output})
        return "root cause: stubbed"

    monkeypatch.setattr(LLMHelper, "analyze_error", fake_analyze)
    return calls


class TestCodeInterpreter:
    async def test_runs_python_and_captures_stdout(self, tools):
        result = await tools.code_interpreter(
            code='print("Test successful")\nprint(f"2 + 2 = {2 + 2}")'
        )

        assert result["success"], result.get("error")
        assert "Test successful" in result["stdout"]
        assert "2 + 2 = 4" in result["stdout"]

    async def test_runtime_failure_reports_error_analysis(self, tools, analysis_stub):
        result = await tools.code_interpreter(code="x = 1 / 0")

        assert result["success"] is False
        assert result["error_analysis"] == "root cause: stubbed"
        assert "ZeroDivisionError" in analysis_stub[0]["error"]

    async def test_syntax_error_is_caught_before_execution(self, tools):
        result = await tools.code_interpreter(code='print("Unclosed string')

        assert result["success"] is False
        assert result["verification"] == "failed"
        assert "Syntax error" in result["error"]

    async def test_successful_run_has_no_error_analysis(self, tools, analysis_stub):
        result = await tools.code_interpreter(code="print('fine')")

        assert result["success"] is True
        assert "error_analysis" not in result
        assert analysis_stub == []

    async def test_error_analysis_can_be_disabled(self, tools, analysis_stub, monkeypatch):
        monkeypatch.setattr(Config, "AUTO_ANALYZE_ERRORS", False)

        result = await tools.code_interpreter(code="x = 1 / 0")

        assert result["success"] is False
        assert "error_analysis" not in result
        assert analysis_stub == []


class TestVirtualTerminal:
    async def test_runs_command_and_captures_stdout(self, tools):
        result = await tools.virtual_terminal(command='echo "Terminal test"')

        assert result["success"], result.get("error")
        assert "Terminal test" in result["stdout"]

    async def test_failing_command_reports_error_analysis(self, tools, analysis_stub):
        result = await tools.virtual_terminal(command="ls /nonexistent_directory_12345")

        assert result["success"] is False
        assert result["returncode"] != 0
        assert result["error_analysis"] == "root cause: stubbed"
        assert analysis_stub[0]["command"] == "ls /nonexistent_directory_12345"

    async def test_successful_command_has_no_error_analysis(self, tools, analysis_stub):
        result = await tools.virtual_terminal(command="echo ok")

        assert result["success"] is True
        assert "error_analysis" not in result
        assert analysis_stub == []

    async def test_timeout_is_reported(self, tools):
        result = await tools.virtual_terminal(command="sleep 5", timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"]

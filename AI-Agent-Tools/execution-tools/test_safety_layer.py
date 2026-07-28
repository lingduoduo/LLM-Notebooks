"""Tests for the approval and verification safety layer.

Regression coverage for the bypasses found in the execution tools:

* Language aliases (``python3``, ``js``, ``sh``, ...) skipped the dangerous
  pattern gate entirely, because the pattern table was keyed on canonical
  names only. ``code_interpreter(language="python3", code="os.system(...)")``
  executed with no approval request at all.
* Only ``python``, ``bash`` and ``php`` had pattern tables, so JavaScript, Go,
  Java, C++, Rust and TypeScript ran unreviewed.
* Substring matching missed trivial spacing and flag-order variants of the
  destructive commands it claimed to catch.
"""

import pytest

from config import Config
from execution_tools import ExecutionTools
from llm_helper import LLMHelper
from multilang_executor import normalize_language


@pytest.fixture
def approvals(monkeypatch):
    """Capture approval requests and deny them all."""
    requests = []

    def fake_request(self, operation, details):
        requests.append({"operation": operation, "details": details})
        return False, "denied by test"

    monkeypatch.setattr(LLMHelper, "request_approval", fake_request)
    monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)
    monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)
    monkeypatch.setattr(Config, "AUTO_ANALYZE_ERRORS", False)
    return requests


@pytest.fixture
def tools():
    return ExecutionTools(LLMHelper())


class TestLanguageNormalization:
    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("python", "python"),
            ("python3", "python"),
            ("PYTHON3", "python"),
            ("js", "javascript"),
            ("node", "javascript"),
            ("nodejs", "javascript"),
            ("ts", "typescript"),
            ("c++", "cpp"),
            ("sh", "bash"),
            ("shell", "bash"),
            (None, "python"),
        ],
    )
    def test_aliases_map_to_canonical_names(self, alias, canonical):
        assert normalize_language(alias) == canonical

    def test_unknown_language_is_passed_through_lowercased(self):
        assert normalize_language("Brainfuck") == "brainfuck"


class TestApprovalCoversEveryAlias:
    @pytest.mark.parametrize("language", ["python", "python3", "PYTHON3"])
    async def test_python_aliases_all_require_approval(self, tools, approvals, language):
        result = await tools.code_interpreter(
            code="import os\nos.system('echo pwned')\n", language=language
        )

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert len(approvals) == 1

    @pytest.mark.parametrize("language", ["bash", "sh", "shell"])
    async def test_shell_aliases_all_require_approval(self, tools, approvals, language):
        result = await tools.code_interpreter(
            code="rm -rf /tmp/whatever", language=language
        )

        assert result["success"] is False
        assert len(approvals) == 1

    @pytest.mark.parametrize("language", ["javascript", "js", "node"])
    async def test_javascript_destructive_calls_require_approval(
        self, tools, approvals, language
    ):
        result = await tools.code_interpreter(
            code="const fs = require('fs');\nfs.rmSync('/tmp/x', {recursive: true});",
            language=language,
        )

        assert result["success"] is False
        assert len(approvals) == 1

    async def test_go_process_execution_requires_approval(self, tools, approvals):
        result = await tools.code_interpreter(
            code='package main\nimport "os/exec"\nfunc main() { exec.Command("rm").Run() }',
            language="go",
        )

        assert result["success"] is False
        assert len(approvals) == 1

    async def test_harmless_code_is_not_gated(self, tools, approvals):
        result = await tools.code_interpreter(code="print(2 + 2)", language="python3")

        assert result["success"] is True
        assert "4" in result["stdout"]
        assert approvals == []


class TestTerminalPatternMatching:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/target",
            "rm -fr /tmp/target",
            "rm  -rf  /tmp/target",
            "rm -r -f /tmp/target",
            "sudo rm -rf /tmp/target",
            "echo hi && rm -rf /tmp/target",
            "mkfs.ext4 /dev/sdb",
            "dd if=/dev/zero of=/dev/sda",
            "chmod -R 777 /",
        ],
    )
    async def test_destructive_command_variants_require_approval(
        self, tools, approvals, command
    ):
        result = await tools.virtual_terminal(command=command)

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert len(approvals) == 1

    @pytest.mark.parametrize(
        "command",
        [
            "echo ok",
            "ls -la",
            "grep -r pattern .",
            "git rm --cached file.txt",
        ],
    )
    async def test_ordinary_commands_run_without_approval(self, tools, approvals, command):
        await tools.virtual_terminal(command=command)

        assert approvals == []


class TestSyntaxVerification:
    """Verification reports three honest states: valid, invalid, unverified.

    It previously returned a bare bool and answered True whenever the LLM call
    raised, so an unreachable verifier was indistinguishable from a clean bill
    of health.
    """

    @pytest.mark.parametrize("language", ["python", "python3"])
    def test_python_aliases_use_local_compilation(self, monkeypatch, language):
        """No alias may fall through to the LLM path for Python."""
        helper = LLMHelper()

        def explode(self):
            raise AssertionError("verification must not contact an LLM for Python")

        monkeypatch.setattr(LLMHelper, "_ensure_client", explode)

        assert helper.verify_code_syntax("x = 1", language) == ("valid", None)

        status, message = helper.verify_code_syntax("def broken(:", language)
        assert status == "invalid"
        assert "Syntax error" in message

    def test_unreachable_verifier_reports_unverified_not_valid(self):
        """An unreachable verifier must never be reported as a passing check."""
        helper = LLMHelper()

        status, message = helper.verify_code_syntax("func main() {", "go")

        assert status == "unverified"
        assert message

    async def test_unverified_language_still_runs_and_says_so(self, tools, monkeypatch):
        """Offline multi-language execution stays available, honestly labelled."""
        monkeypatch.setattr(Config, "AUTO_VERIFY_CODE", True)
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)
        monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)

        result = await tools.code_interpreter(code="echo hello", language="bash")

        assert result["success"] is True
        assert "hello" in result["stdout"]
        assert result["verification"] == "unverified"

    async def test_python_execution_reports_passed_verification(self, tools, monkeypatch):
        monkeypatch.setattr(Config, "AUTO_VERIFY_CODE", True)
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)
        monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)

        result = await tools.code_interpreter(code="print('ok')", language="python")

        assert result["success"] is True
        assert result["verification"] == "passed"

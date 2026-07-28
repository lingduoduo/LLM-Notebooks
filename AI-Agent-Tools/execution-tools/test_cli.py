"""Tests for the command line entry point.

The CLI used to keep its own hard-coded TOOL_CATALOG alongside the schemas in
server.py, so the two could describe different tool sets. It now builds from
the shared registry, and these tests hold that alignment in place.
"""

import pytest

import cli
from config import Config
from tool_registry import build_registry


@pytest.fixture(autouse=True)
def offline_cli(workspace, monkeypatch):
    """Run the CLI offline, against the isolated workspace."""
    monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)
    monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)
    monkeypatch.setattr(Config, "AUTO_ANALYZE_ERRORS", False)
    return workspace


class TestListCommand:
    def test_lists_every_registered_tool(self, capsys):
        exit_code = cli.main(["list"])

        out = capsys.readouterr().out
        assert exit_code == 0
        for name in build_registry():
            assert name in out

    def test_reports_the_tool_count(self, capsys):
        cli.main(["list"])

        out = capsys.readouterr().out
        assert f"{len(build_registry())} tools registered." in out


class TestCodeCommand:
    def test_executes_python(self, capsys):
        exit_code = cli.main(["code", "--code", "print(2 ** 10)"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "1024" in out

    def test_reports_failure_with_a_nonzero_exit_code(self, capsys):
        exit_code = cli.main(["code", "--code", "raise SystemExit(3)"])

        assert exit_code == 1

    def test_missing_code_is_an_argument_error(self, capsys):
        exit_code = cli.main(["code"])

        assert exit_code == 2
        assert "--code or --file" in capsys.readouterr().err

    def test_language_alias_is_accepted(self, capsys):
        exit_code = cli.main(["code", "--language", "python3", "--code", "print('aliased')"])

        assert exit_code == 0
        assert "aliased" in capsys.readouterr().out


class TestShellCommand:
    def test_runs_a_command(self, capsys):
        exit_code = cli.main(["shell", "echo cli-shell-test"])

        assert exit_code == 0
        assert "cli-shell-test" in capsys.readouterr().out


class TestWriteAndEditCommands:
    def test_write_then_edit(self, capsys, workspace):
        assert cli.main(["write", "--path", "notes.txt", "--content", "hello"]) == 0
        assert cli.main(["edit", "--path", "notes.txt", "--search", "hello",
                         "--replace", "world"]) == 0

        assert (workspace / "notes.txt").read_text() == "world"

    def test_write_refuses_to_clobber_without_overwrite(self, capsys, workspace):
        (workspace / "notes.txt").write_text("ORIGINAL")

        exit_code = cli.main(["write", "--path", "notes.txt", "--content", "NEW"])

        assert exit_code == 1
        assert (workspace / "notes.txt").read_text() == "ORIGINAL"

    def test_overwrite_flag_allows_replacement(self, capsys, workspace):
        (workspace / "notes.txt").write_text("ORIGINAL")

        exit_code = cli.main(
            ["--no-approval", "write", "--path", "notes.txt",
             "--content", "NEW", "--overwrite"]
        )

        assert exit_code == 0
        assert (workspace / "notes.txt").read_text() == "NEW"

    def test_missing_content_is_an_argument_error(self, capsys):
        exit_code = cli.main(["write", "--path", "x.txt"])

        assert exit_code == 2
        assert "--content or --content-file" in capsys.readouterr().err


class TestGlobalSwitches:
    def test_provider_option_is_gone(self):
        """The provider router was replaced by direct OpenAI configuration."""
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--provider", "kimi", "list"])

    def test_no_approval_disables_the_gate(self, monkeypatch):
        monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)

        cli.main(["--no-approval", "shell", "echo ok"])

        assert Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS is False

    def test_workspace_switch_redirects_file_operations(self, tmp_path, capsys):
        target = tmp_path / "elsewhere"
        target.mkdir()

        cli.main(["--workspace", str(target), "write", "--path", "a.txt",
                  "--content", "in-elsewhere"])

        assert (target / "a.txt").read_text() == "in-elsewhere"


class TestDemo:
    def test_demo_runs_offline_to_completion(self, capsys):
        exit_code = cli.main(["demo"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "Demo complete" in out
        # Each stage must have produced output, not silently no-opped.
        for stage in ("1. file_write", "5. code_interpreter",
                      "7. code_interpreter", "9. virtual_terminal"):
            assert stage in out

    def test_demo_output_is_english(self, capsys):
        import re

        cli.main(["demo"])

        out = capsys.readouterr().out
        assert not re.search("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", out)

"""Tests for the shared tool registry.

`filesystem_enhanced` and `terminal_controller` were unreachable: roughly
1,200 lines implementing ~20 operations that no MCP client could call because
neither module was referenced by `server.py` or `cli.py`. They are now
registered through one registry that both entry points consume, so the tool
list cannot drift between them.

Registering them also means their operations must pass the same safety layer
as the rest: `TerminalController.execute_command` was an unguarded shell and
`FilesystemEnhanced.delete_file(recursive=True)` could erase the workspace,
neither with any approval step.
"""

import pytest

from config import Config
from llm_helper import LLMHelper
from tool_registry import build_registry


@pytest.fixture
def registry(workspace):
    return build_registry()


@pytest.fixture
def approvals(monkeypatch):
    requests = []

    def fake_request(self, operation, details):
        requests.append({"operation": operation, "details": details})
        return False, "denied by test"

    monkeypatch.setattr(LLMHelper, "request_approval", fake_request)
    monkeypatch.setattr(Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", True)
    monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)
    monkeypatch.setattr(Config, "AUTO_ANALYZE_ERRORS", False)
    return requests


EXPECTED_TOOLS = {
    # Original surface
    "file_write",
    "file_edit",
    "code_interpreter",
    "virtual_terminal",
    "google_calendar_add",
    "github_create_pr",
    # Enhanced filesystem
    "fs_read_file",
    "fs_read_multiple_files",
    "fs_list_directory",
    "fs_directory_tree",
    "fs_search_files",
    "fs_get_file_info",
    "fs_move",
    "fs_copy",
    "fs_delete",
    "fs_create_directory",
    "fs_list_allowed_directories",
    # Stateful terminal session
    "terminal_execute",
    "terminal_pwd",
    "terminal_cd",
    "terminal_insert_lines",
    "terminal_delete_lines",
    "terminal_update_line",
    "terminal_history",
}


class TestRegistryShape:
    def test_every_expected_tool_is_registered(self, registry):
        assert EXPECTED_TOOLS <= set(registry)

    def test_each_tool_has_a_schema_and_handler(self, registry):
        for name, spec in registry.items():
            assert spec.description, f"{name} has no description"
            assert spec.input_schema["type"] == "object", name
            assert callable(spec.handler), name

    def test_required_properties_exist_in_the_schema(self, registry):
        for name, spec in registry.items():
            properties = spec.input_schema.get("properties", {})
            for required in spec.input_schema.get("required", []):
                assert required in properties, f"{name}: {required} not described"


class TestFilesystemToolsAreReachable:
    async def test_read_file_round_trips(self, registry, workspace):
        (workspace / "note.txt").write_text("hello enhanced")

        result = await registry["fs_read_file"].handler(file_path="note.txt")

        assert result["success"] is True
        assert result["content"] == "hello enhanced"

    async def test_search_files_finds_matches(self, registry, workspace):
        (workspace / "a.py").write_text("x = 1")
        (workspace / "b.txt").write_text("text")

        result = await registry["fs_search_files"].handler(pattern="*.py")

        assert result["success"] is True
        assert [entry["path"] for entry in result["files"]] == ["a.py"]

    async def test_create_directory_works(self, registry, workspace):
        result = await registry["fs_create_directory"].handler(directory_path="sub/dir")

        assert result["success"] is True
        assert (workspace / "sub" / "dir").is_dir()


class TestDestructiveFilesystemToolsRequireApproval:
    async def test_recursive_delete_is_gated(self, registry, workspace, approvals):
        target = workspace / "victim"
        target.mkdir()
        (target / "keep.txt").write_text("precious")

        result = await registry["fs_delete"].handler(
            file_path="victim", recursive=True
        )

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert (target / "keep.txt").read_text() == "precious"
        assert approvals[0]["operation"] == "file_delete"

    async def test_move_over_an_existing_file_is_gated(self, registry, workspace, approvals):
        (workspace / "src.txt").write_text("new")
        (workspace / "dst.txt").write_text("existing")

        result = await registry["fs_move"].handler(
            source="src.txt", destination="dst.txt", overwrite=True
        )

        assert result["success"] is False
        assert (workspace / "dst.txt").read_text() == "existing"

    async def test_plain_delete_of_a_single_file_is_gated(self, registry, workspace, approvals):
        (workspace / "one.txt").write_text("data")

        result = await registry["fs_delete"].handler(file_path="one.txt")

        assert result["success"] is False
        assert (workspace / "one.txt").exists()


class TestTerminalSessionTools:
    async def test_execute_runs_in_the_session_directory(self, registry, workspace):
        (workspace / "sub").mkdir()

        await registry["terminal_cd"].handler(directory="sub")
        pwd = await registry["terminal_pwd"].handler()
        result = await registry["terminal_execute"].handler(command="pwd")

        assert pwd["current_directory"].endswith("sub")
        assert result["success"] is True
        assert result["stdout"].strip().endswith("sub")

    async def test_dangerous_command_requires_approval(self, registry, approvals):
        result = await registry["terminal_execute"].handler(
            command="rm -rf /tmp/anything"
        )

        assert result["success"] is False
        assert "not approved" in result["error"]
        assert approvals[0]["operation"] == "terminal_command"

    async def test_long_output_is_truncated_and_persisted(self, registry, monkeypatch):
        monkeypatch.setattr(Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)

        result = await registry["terminal_execute"].handler(
            command="for i in $(seq 1 500); do echo line$i; done"
        )

        assert result["success"] is True
        assert len(result["stdout"].splitlines()) < 200
        assert result["stdout_file"] is not None
        assert "line500" in open(result["stdout_file"], encoding="utf-8").read()

    async def test_history_is_recorded(self, registry):
        await registry["terminal_execute"].handler(command="echo one")
        await registry["terminal_execute"].handler(command="echo two")

        result = await registry["terminal_history"].handler(count=2)

        assert result["history"] == ["echo one", "echo two"]

    async def test_line_editing_tools_are_reachable(self, registry, workspace):
        target = workspace / "lines.txt"
        target.write_text("a\nb\nc\n")

        await registry["terminal_insert_lines"].handler(
            file_path="lines.txt", content="inserted", line_number=2
        )
        await registry["terminal_update_line"].handler(
            file_path="lines.txt", line_number=1, new_content="A"
        )
        result = await registry["terminal_delete_lines"].handler(
            file_path="lines.txt", start_line=4, end_line=4
        )

        assert result["success"] is True
        assert target.read_text() == "A\ninserted\nb\n"

    async def test_session_stays_inside_the_workspace(self, registry):
        result = await registry["terminal_cd"].handler(directory="/etc")

        assert result["success"] is False
        assert "outside workspace" in result["error"]

"""Single source of truth for the execution tools exposed to clients.

`server.py` previously hard-coded the MCP schemas while `cli.py` kept its own
parallel catalog, so the two could -- and did -- describe different tool sets.
Both now build from this registry, which also brings the previously
unreachable `filesystem_enhanced` and `terminal_controller` modules into the
tool surface.

Deliberately not registered: `TerminalController.read_file`, `write_file` and
`list_directory`. They duplicate `fs_read_file`, `file_write` and
`fs_list_directory` exactly, and offering an agent two names for one operation
degrades tool selection without adding capability. The session-stateful and
line-level operations that nothing else provides are registered.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from execution_tools import ExecutionTools
from external_tools import ExternalTools
from file_tools import FileTools
from filesystem_enhanced import FilesystemEnhanced
from llm_helper import LLMHelper
from terminal_controller import TerminalController

STRING = {"type": "string"}


@dataclass
class ToolSpec:
    """One callable tool: how it is described and how it is invoked."""

    name: str
    category: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Awaitable[Dict[str, Any]]] = field(repr=False)


def _schema(properties: Dict[str, Any], required=()) -> Dict[str, Any]:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    return schema


def build_registry(llm_helper: Optional[LLMHelper] = None) -> Dict[str, ToolSpec]:
    """Instantiate every tool and return it keyed by tool name.

    Tools read ``Config.WORKSPACE_DIR`` when constructed, so build the registry
    after the workspace has been configured.
    """
    llm_helper = llm_helper or LLMHelper()

    files = FileTools(llm_helper)
    execution = ExecutionTools(llm_helper)
    external = ExternalTools(llm_helper)
    enhanced = FilesystemEnhanced(llm_helper)
    terminal = TerminalController(llm_helper)

    specs = [
        # --- File system -------------------------------------------------
        ToolSpec(
            name="file_write",
            category="file system",
            description="Write content to a file with automatic syntax verification",
            input_schema=_schema(
                {
                    "path": {**STRING, "description": "File path (relative to workspace or absolute)"},
                    "content": {**STRING, "description": "Content to write"},
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace the file if it already exists",
                        "default": False,
                    },
                },
                required=("path", "content"),
            ),
            handler=files.write_file,
        ),
        ToolSpec(
            name="file_edit",
            category="file system",
            description="Edit an existing file by searching and replacing content",
            input_schema=_schema(
                {
                    "path": {**STRING, "description": "File path"},
                    "search": {**STRING, "description": "Text to search for"},
                    "replace": {**STRING, "description": "Replacement text"},
                },
                required=("path", "search", "replace"),
            ),
            handler=files.edit_file,
        ),
        # --- General execution -------------------------------------------
        ToolSpec(
            name="code_interpreter",
            category="execution",
            description=(
                "Execute code in a temporary directory with syntax verification, "
                "approval for dangerous operations, and long-output handling. "
                "Supports: Python, JavaScript, TypeScript, Go, Java, C++, Rust, PHP, Bash"
            ),
            input_schema=_schema(
                {
                    "code": {**STRING, "description": "Code to execute"},
                    "language": {
                        **STRING,
                        "description": "Programming language (python, javascript, typescript, go, java, cpp, rust, php, bash)",
                        "default": "python",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Execution timeout in seconds",
                        "default": 30.0,
                    },
                    "stdin": {**STRING, "description": "Optional stdin input for the program"},
                    "files": {
                        "type": "object",
                        "description": (
                            "Optional additional files (filename -> content). "
                            "Content is written as UTF-8 text unless it carries "
                            "the explicit 'base64:' prefix."
                        ),
                        "additionalProperties": STRING,
                    },
                },
                required=("code",),
            ),
            handler=execution.code_interpreter,
        ),
        ToolSpec(
            name="virtual_terminal",
            category="execution",
            description="Execute a shell command with approval for dangerous commands and long-output handling",
            input_schema=_schema(
                {
                    "command": {**STRING, "description": "Shell command to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                required=("command",),
            ),
            handler=execution.virtual_terminal,
        ),
        # --- External systems --------------------------------------------
        ToolSpec(
            name="google_calendar_add",
            category="external system",
            description="Add an event to Google Calendar",
            input_schema=_schema(
                {
                    "summary": {**STRING, "description": "Event title"},
                    "start_time": {**STRING, "description": "Start time (ISO 8601, e.g. 2026-01-01T10:00:00)"},
                    "end_time": {**STRING, "description": "End time (ISO 8601)"},
                    "description": {**STRING, "description": "Event description"},
                    "location": {**STRING, "description": "Event location"},
                },
                required=("summary", "start_time", "end_time"),
            ),
            handler=external.google_calendar_add,
        ),
        ToolSpec(
            name="github_create_pr",
            category="external system",
            description="Create a GitHub pull request",
            input_schema=_schema(
                {
                    "repo_name": {**STRING, "description": "Repository name (owner/repo)"},
                    "title": {**STRING, "description": "PR title"},
                    "body": {**STRING, "description": "PR description"},
                    "head_branch": {**STRING, "description": "Source branch"},
                    "base_branch": {**STRING, "description": "Target branch", "default": "main"},
                },
                required=("repo_name", "title", "body", "head_branch"),
            ),
            handler=external.github_create_pr,
        ),
        # --- Enhanced file system ----------------------------------------
        ToolSpec(
            name="fs_read_file",
            category="file system",
            description="Read a text file, with an enforced size limit",
            input_schema=_schema(
                {
                    "file_path": {**STRING, "description": "Path to the file"},
                    "encoding": {**STRING, "description": "File encoding", "default": "utf-8"},
                    "max_size_mb": {
                        "type": "integer",
                        "description": "Maximum file size in MB",
                        "default": 10,
                    },
                },
                required=("file_path",),
            ),
            handler=enhanced.read_text_file,
        ),
        ToolSpec(
            name="fs_read_multiple_files",
            category="file system",
            description="Read several text files in one call",
            input_schema=_schema(
                {
                    "file_paths": {
                        "type": "array",
                        "items": STRING,
                        "description": "Paths to read",
                    },
                    "encoding": {**STRING, "description": "File encoding", "default": "utf-8"},
                },
                required=("file_paths",),
            ),
            handler=enhanced.read_multiple_files,
        ),
        ToolSpec(
            name="fs_list_directory",
            category="file system",
            description="List directory contents with per-entry and total sizes",
            input_schema=_schema(
                {"directory_path": {**STRING, "description": "Directory to list", "default": "."}}
            ),
            handler=enhanced.list_directory_with_sizes,
        ),
        ToolSpec(
            name="fs_directory_tree",
            category="file system",
            description="Render a directory tree up to a maximum depth",
            input_schema=_schema(
                {
                    "directory_path": {**STRING, "description": "Root directory", "default": "."},
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse",
                        "default": 3,
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include dot files",
                        "default": False,
                    },
                }
            ),
            handler=enhanced.directory_tree,
        ),
        ToolSpec(
            name="fs_search_files",
            category="file system",
            description="Find files matching a glob pattern",
            input_schema=_schema(
                {
                    "pattern": {**STRING, "description": "Glob pattern, e.g. *.py"},
                    "directory_path": {**STRING, "description": "Directory to search", "default": "."},
                    "recursive": {
                        "type": "boolean",
                        "description": "Search subdirectories",
                        "default": True,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive matching",
                        "default": False,
                    },
                },
                required=("pattern",),
            ),
            handler=enhanced.search_files,
        ),
        ToolSpec(
            name="fs_get_file_info",
            category="file system",
            description="Report size, timestamps, permissions and type for a path",
            input_schema=_schema(
                {"file_path": {**STRING, "description": "Path to inspect"}},
                required=("file_path",),
            ),
            handler=enhanced.get_file_info,
        ),
        ToolSpec(
            name="fs_move",
            category="file system",
            description="Move or rename a file or directory (approval required to replace an existing destination)",
            input_schema=_schema(
                {
                    "source": {**STRING, "description": "Source path"},
                    "destination": {**STRING, "description": "Destination path"},
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace an existing destination",
                        "default": False,
                    },
                },
                required=("source", "destination"),
            ),
            handler=enhanced.move_file,
        ),
        ToolSpec(
            name="fs_copy",
            category="file system",
            description="Copy a file or directory (approval required to replace an existing destination)",
            input_schema=_schema(
                {
                    "source": {**STRING, "description": "Source path"},
                    "destination": {**STRING, "description": "Destination path"},
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace an existing destination",
                        "default": False,
                    },
                },
                required=("source", "destination"),
            ),
            handler=enhanced.copy_file,
        ),
        ToolSpec(
            name="fs_delete",
            category="file system",
            description="Delete a file or directory (approval required)",
            input_schema=_schema(
                {
                    "file_path": {**STRING, "description": "Path to delete"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Required to delete a directory",
                        "default": False,
                    },
                },
                required=("file_path",),
            ),
            handler=enhanced.delete_file,
        ),
        ToolSpec(
            name="fs_create_directory",
            category="file system",
            description="Create a new directory",
            input_schema=_schema(
                {
                    "directory_path": {**STRING, "description": "Directory to create"},
                    "parents": {
                        "type": "boolean",
                        "description": "Create missing parent directories",
                        "default": True,
                    },
                },
                required=("directory_path",),
            ),
            handler=enhanced.create_directory,
        ),
        ToolSpec(
            name="fs_list_allowed_directories",
            category="file system",
            description="List the directories file operations are confined to",
            input_schema=_schema({}),
            handler=enhanced.list_allowed_directories,
        ),
        # --- Stateful terminal session ------------------------------------
        ToolSpec(
            name="terminal_execute",
            category="terminal session",
            description=(
                "Run a command in the session's current directory. Unlike "
                "virtual_terminal this keeps a working directory and history "
                "across calls."
            ),
            input_schema=_schema(
                {
                    "command": {**STRING, "description": "Command to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                required=("command",),
            ),
            handler=terminal.execute_command,
        ),
        ToolSpec(
            name="terminal_pwd",
            category="terminal session",
            description="Report the session's current directory",
            input_schema=_schema({}),
            handler=terminal.get_current_directory,
        ),
        ToolSpec(
            name="terminal_cd",
            category="terminal session",
            description="Change the session's current directory (confined to the workspace)",
            input_schema=_schema(
                {"directory": {**STRING, "description": "Directory to change to"}},
                required=("directory",),
            ),
            handler=terminal.change_directory,
        ),
        ToolSpec(
            name="terminal_insert_lines",
            category="terminal session",
            description="Insert content at a specific line of a file (1-indexed)",
            input_schema=_schema(
                {
                    "file_path": {**STRING, "description": "File path"},
                    "content": {**STRING, "description": "Content to insert"},
                    "line_number": {
                        "type": "integer",
                        "description": "Line to insert at (1-indexed)",
                    },
                },
                required=("file_path", "content", "line_number"),
            ),
            handler=terminal.insert_file_content,
        ),
        ToolSpec(
            name="terminal_delete_lines",
            category="terminal session",
            description="Delete an inclusive range of lines from a file (1-indexed)",
            input_schema=_schema(
                {
                    "file_path": {**STRING, "description": "File path"},
                    "start_line": {"type": "integer", "description": "First line to delete"},
                    "end_line": {"type": "integer", "description": "Last line to delete"},
                },
                required=("file_path", "start_line", "end_line"),
            ),
            handler=terminal.delete_file_content,
        ),
        ToolSpec(
            name="terminal_update_line",
            category="terminal session",
            description="Replace a single line of a file (1-indexed)",
            input_schema=_schema(
                {
                    "file_path": {**STRING, "description": "File path"},
                    "line_number": {"type": "integer", "description": "Line to replace"},
                    "new_content": {**STRING, "description": "New content for the line"},
                },
                required=("file_path", "line_number", "new_content"),
            ),
            handler=terminal.update_file_content,
        ),
        ToolSpec(
            name="terminal_history",
            category="terminal session",
            description="List recently attempted commands in this session",
            input_schema=_schema(
                {
                    "count": {
                        "type": "integer",
                        "description": "How many recent commands to return",
                        "default": 10,
                    }
                }
            ),
            handler=terminal.get_command_history,
        ),
    ]

    return {spec.name: spec for spec in specs}

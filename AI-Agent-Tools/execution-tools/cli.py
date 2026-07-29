#!/usr/bin/env python3
"""Unified command line entry point for the execution tools.

This module provides an argparse interface to list the tools, invoke each one
individually, and run an end-to-end offline demo. It builds its tools from the
same `tool_registry` that `server.py` uses, so CLI behavior matches the MCP
server exactly and the two cannot describe different tool sets.

Safety mechanisms (matching the "execution tools" section of the book):
  - LLM pre-approval: irreversible or destructive operations are reviewed by a
    separate LLM before they run
  - Validation: Python syntax is checked locally with compile(); other
    languages are checked by their own compiler or interpreter
  - Long-output truncation and persistence: output over the threshold is
    reduced to a head and tail, with the complete text written to a file

Usage examples:
  python cli.py list
  python cli.py demo
  python cli.py code --language python --code "print(2 ** 10)"
  python cli.py shell "python3 --version"
  python cli.py write --path notes.txt --content "hello" --overwrite
  python cli.py --no-approval --no-summarize shell "ls -la"

Commands that need no API key: list, demo (offline path), and code/shell/
write/edit with approval and summarization disabled. An API key is needed for
LLM pre-approval, summarization of long output, and error analysis. The
calendar and pr commands additionally need their own external credentials.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import textwrap


def _apply_settings(**overrides) -> None:
    """Apply configuration overrides to both the environment and live Config.

    config.Config reads the environment at import time. Setting only the
    environment works for a fresh process but silently does nothing once the
    module has been imported, so each override is also written to the loaded
    class.
    """
    from pathlib import Path

    from config import Config

    for name, value in overrides.items():
        if value is None:
            continue
        os.environ[name] = str(value).lower() if isinstance(value, bool) else str(value)
        setattr(Config, name, Path(value) if name == "WORKSPACE_DIR" else value)


def _apply_global_env(args: argparse.Namespace) -> None:
    """Copy global switches into the configuration."""
    _apply_settings(
        WORKSPACE_DIR=os.path.abspath(args.workspace) if args.workspace else None,
        REQUIRE_APPROVAL_FOR_DANGEROUS_OPS=False if args.no_approval else None,
        AUTO_VERIFY_CODE=False if args.no_verify else None,
        AUTO_SUMMARIZE_COMPLEX_OUTPUT=False if args.no_summarize else None,
    )


def _registry():
    """Build the shared tool registry (imported late so env vars apply)."""
    from tool_registry import build_registry

    return build_registry()


def _print_result(result: dict) -> None:
    """Print a tool result as JSON."""
    print(json.dumps(result, indent=2, default=str))


def _run(coro) -> int:
    """Run a tool coroutine, print its result, and derive an exit code."""
    result = asyncio.run(coro)
    _print_result(result)
    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    registry = _registry()

    print("Available execution tools:\n")
    print(f"  {'Tool':<28} {'Category':<18} Description")
    print(f"  {'-' * 28} {'-' * 18} {'-' * 40}")
    for spec in registry.values():
        description = spec.description
        if len(description) > 60:
            description = description[:57] + "..."
        print(f"  {spec.name:<28} {spec.category:<18} {description}")

    print(f"\n{len(registry)} tools registered.")
    print("Use `python cli.py <subcommand> --help` to view each tool's arguments.")
    print("Use `python cli.py demo` to run the end-to-end offline demo.")
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    code = args.code
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    if not code:
        print("Error: provide code with --code or --file.", file=sys.stderr)
        return 2

    return _run(_registry()["code_interpreter"].handler(
        code=code,
        language=args.language,
        timeout=args.timeout,
        stdin=args.stdin,
    ))


def cmd_shell(args: argparse.Namespace) -> int:
    return _run(_registry()["virtual_terminal"].handler(
        command=args.command,
        timeout=args.timeout,
    ))


def cmd_write(args: argparse.Namespace) -> int:
    content = args.content
    if args.content_file:
        with open(args.content_file, "r", encoding="utf-8") as f:
            content = f.read()
    if content is None:
        print("Error: provide file content with --content or --content-file.",
              file=sys.stderr)
        return 2

    return _run(_registry()["file_write"].handler(
        path=args.path,
        content=content,
        overwrite=args.overwrite,
    ))


def cmd_edit(args: argparse.Namespace) -> int:
    return _run(_registry()["file_edit"].handler(
        path=args.path,
        search=args.search,
        replace=args.replace,
    ))


def cmd_calendar(args: argparse.Namespace) -> int:
    return _run(_registry()["google_calendar_add"].handler(
        summary=args.summary,
        start_time=args.start,
        end_time=args.end,
        description=args.description,
        location=args.location,
    ))


def cmd_pr(args: argparse.Namespace) -> int:
    return _run(_registry()["github_create_pr"].handler(
        repo_name=args.repo,
        title=args.title,
        body=args.body,
        head_branch=args.head,
        base_branch=args.base,
    ))


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end offline demo: an agent completing a small real task.

    Scenario: the agent writes a word-frequency script, generates sample data,
    runs the statistics, and verifies the result with the shell. Along the way
    it exercises validation, dangerous-command approval, and long-output
    truncation and persistence. The whole flow runs offline by default.
    """
    # Run in an isolated temporary workspace so the demo cannot touch the
    # current directory.
    workspace = tempfile.mkdtemp(prefix="exec_tools_demo_")
    # Offline: disable the LLM-backed steps. Truncation and persistence do not
    # need an LLM.
    _apply_settings(
        WORKSPACE_DIR=workspace,
        AUTO_SUMMARIZE_COMPLEX_OUTPUT=False,
        AUTO_ANALYZE_ERRORS=False,
    )

    registry = _registry()
    write_file = registry["file_write"].handler
    edit_file = registry["file_edit"].handler
    run_code = registry["code_interpreter"].handler
    run_shell = registry["virtual_terminal"].handler
    read_file = registry["fs_read_file"].handler

    def section(title: str) -> None:
        print("\n" + "=" * 64)
        print(title)
        print("=" * 64)

    print(f"Demo workspace: {workspace}")
    print("(Offline path, no API key required. With a key configured, the "
          "approval and summarization steps call a real LLM.)")

    async def run() -> None:
        # 1. Write a file, validated automatically before it lands
        section("1. file_write: write the word-frequency script (validated)")
        script = textwrap.dedent('''\
            """Count word frequencies in a text file."""
            import sys
            from collections import Counter

            def word_count(path):
                with open(path, encoding="utf-8") as f:
                    words = f.read().split()
                return Counter(words)

            if __name__ == "__main__":
                for word, freq in word_count(sys.argv[1]).most_common(5):
                    print(f"{word}\\t{freq}")
            ''')
        r = await write_file(path="wordcount.py", content=script, overwrite=True)
        print(f"Result: success={r['success']}, verification={r.get('verification')}")
        print(f"Wrote: {r.get('path')}")

        # 2. Validation rejects broken code
        section("2. file_write: write code with a syntax error (must be rejected)")
        r = await write_file(path="broken.py", content="def broken(:\n    return 1\n",
                             overwrite=True)
        print(f"Result: success={r['success']}")
        print(f"Validation feedback: {r.get('error')}")

        # 3. An existing file is never replaced without an explicit request
        section("3. file_write: overwrite=False must not clobber an existing file")
        r = await write_file(path="wordcount.py", content="# replaced\n")
        print(f"Result: success={r['success']}")
        print(f"Refusal: {r.get('error')}")

        # 4. Generate sample data
        section("4. file_write: generate the sample data file")
        sample = "apple banana apple cherry banana apple date cherry banana apple\n"
        r = await write_file(path="data.txt", content=sample, overwrite=True)
        print(f"Result: success={r['success']}, wrote {r.get('bytes_written')} bytes")

        # 5. Run the statistics
        section("5. code_interpreter: run the statistics")
        analysis = textwrap.dedent('''\
            from collections import Counter
            text = "apple banana apple cherry banana apple date cherry banana apple"
            for word, freq in Counter(text.split()).most_common(3):
                print(f"{word}: {freq}")
            ''')
        r = await run_code(code=analysis, language="python")
        print(f"Result: success={r['success']}, returncode={r.get('returncode')}")
        print("stdout:")
        print(textwrap.indent(r.get("stdout", ""), "  "))

        # 6. Verify the data with the shell
        section("6. virtual_terminal: verify the data file with the shell")
        r = await run_shell(command=f"wc -w {workspace}/data.txt && echo '--- word count done ---'")
        print(f"Result: success={r['success']}, returncode={r.get('returncode')}")
        print("stdout:")
        print(textwrap.indent(r.get("stdout", ""), "  "))

        # 7. Long output is truncated for context and persisted in full
        section("7. code_interpreter: truncate and persist long output")
        long_code = "for i in range(1000):\n    print(f'line {i}: ' + 'x' * 20)\n"
        r = await run_code(code=long_code, language="python")
        stdout = r.get("stdout", "")
        print(f"Lines kept in context: {len(stdout.splitlines())} (of 1000 produced)")
        print(f"Complete output saved to: {r.get('stdout_file')}")
        print("Tail of the retained output:")
        print(textwrap.indent("\n".join(stdout.splitlines()[-4:]), "  "))

        # 8. The saved output is readable through the filesystem tools
        section("8. fs_read_file: read the persisted output back")
        saved = r.get("stdout_file")
        if saved:
            back = await read_file(file_path=saved, max_size_mb=10)
            if back["success"]:
                print(f"Read back {back['lines']} lines from the saved file.")
            else:
                print(f"Saved output is outside the workspace: {back['error']}")
                print("(Expected: persisted output lives in a process-owned "
                      "temporary directory, not in the workspace.)")

        # 9. Dangerous commands need approval
        section("9. virtual_terminal: a dangerous command requires approval")
        _apply_settings(REQUIRE_APPROVAL_FOR_DANGEROUS_OPS=True)
        # The target is a non-existent temporary path, so it is harmless even
        # if it were to run.
        danger = await run_shell(command="rm -rf /tmp/exec_tools_demo_nonexistent_path_xyz")
        print(f"Result: success={danger['success']}")
        if danger.get("error"):
            print(f"Detail: {danger.get('error')}")
            print("(Approval refused: the command was blocked and never ran. "
                  "Offline with no LLM the check fails safe and refuses; "
                  "online a real LLM may also judge it too risky.)")
        else:
            print("(Approved: an API key is configured and the LLM judged this "
                  "command harmless because the path does not exist.)")

        section("Demo complete")
        print("Mechanisms covered: automatic validation, overwrite protection, "
              "dangerous-command approval, long-output truncation and persistence.")
        print(f"Demo artifacts are in: {workspace}")

    asyncio.run(run())
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Unified command line entry point for the execution tools (experiment 4-2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python cli.py list                      list every execution tool
              python cli.py demo                      run the end-to-end offline demo
              python cli.py code --code "print(6*7)"  execute Python code
              python cli.py shell "ls -la"            execute a shell command
              python cli.py write --path a.txt --content hi --overwrite
              python cli.py --no-approval shell "echo hello"

            With --no-approval / --no-summarize / --no-verify, the code, shell,
            write and edit commands run entirely offline without an API key.
        """),
    )

    # Global switches
    parser.add_argument("--workspace",
                        help="Workspace directory (overrides WORKSPACE_DIR; file operations are confined to it)")
    parser.add_argument("--no-approval", action="store_true",
                        help="Disable LLM pre-approval for dangerous operations")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable automatic syntax validation for writes and code")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Disable LLM summarization of long output (still truncates and persists)")

    sub = parser.add_subparsers(dest="command", metavar="<subcommand>")

    p = sub.add_parser("list", help="List every available execution tool")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("demo", help="Run the end-to-end offline demo (start here)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("code", help="Invoke code_interpreter to execute code")
    p.add_argument("--code", help="Code to execute")
    p.add_argument("--file", help="Read the code to execute from a file")
    p.add_argument("--language", default="python",
                   help="Programming language (python/javascript/typescript/go/java/cpp/rust/php/bash; default python)")
    p.add_argument("--timeout", type=float, default=30.0, help="Execution timeout in seconds (default 30)")
    p.add_argument("--stdin", help="Optional standard input")
    p.set_defaults(func=cmd_code)

    p = sub.add_parser("shell", help="Invoke virtual_terminal to execute a shell command")
    p.add_argument("command", help="Shell command to execute")
    p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds (default 30)")
    p.set_defaults(func=cmd_shell)

    p = sub.add_parser("write", help="Invoke file_write to write a file")
    p.add_argument("--path", required=True, help="File path (relative to the workspace, or absolute)")
    p.add_argument("--content", help="File content")
    p.add_argument("--content-file", help="Read the content to write from a file")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing file")
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("edit", help="Invoke file_edit to search and replace in a file")
    p.add_argument("--path", required=True, help="File path")
    p.add_argument("--search", required=True, help="Text to search for")
    p.add_argument("--replace", required=True, help="Replacement text")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("calendar", help="Invoke google_calendar_add to create an event (needs credentials)")
    p.add_argument("--summary", required=True, help="Event title")
    p.add_argument("--start", required=True, help="Start time (ISO 8601, e.g. 2026-10-01T10:00:00)")
    p.add_argument("--end", required=True, help="End time (ISO 8601)")
    p.add_argument("--description", help="Event description")
    p.add_argument("--location", help="Event location")
    p.set_defaults(func=cmd_calendar)

    p = sub.add_parser("pr", help="Invoke github_create_pr to open a pull request (needs a token)")
    p.add_argument("--repo", required=True, help="Repository name (owner/repo)")
    p.add_argument("--title", required=True, help="PR title")
    p.add_argument("--body", required=True, help="PR description")
    p.add_argument("--head", required=True, help="Source branch")
    p.add_argument("--base", default="main", help="Target branch (default main)")
    p.set_defaults(func=cmd_pr)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    _apply_global_env(args)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

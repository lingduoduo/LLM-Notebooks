"""Detection of operations that need approval before they run.

The previous implementation tested raw substrings (``'rm -rf' in command``),
which both over- and under-matched: ``rm -fr`` and ``rm -r -f`` slipped
through, while ``git rm --cached`` and any Python script containing ``open(``
triggered a needless LLM round-trip.

Commands are now tokenized and judged on the actual command word and its
flags; source code is matched with word-boundary regexes keyed on canonical
language names (see ``multilang_executor.normalize_language``), so no alias
can reach an empty pattern table.
"""

import re
import shlex
from typing import List

# Wrappers that precede the real command word and must be skipped.
COMMAND_WRAPPERS = {"sudo", "doas", "env", "nohup", "time", "nice", "xargs"}

# Commands that are destructive regardless of their arguments.
DESTRUCTIVE_COMMANDS = {
    "dd",
    "fdisk",
    "shred",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "mkswap",
}

# Commands that are destructive only with a recursive or force flag.
FLAG_SENSITIVE_COMMANDS = {"rm", "chmod", "chown", "chgrp"}

RECURSIVE_OR_FORCE_FLAGS = {"--recursive", "--force", "--no-preserve-root"}

_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;&|\n]")

# Writing directly to a device node is destructive no matter the command.
_DEVICE_REDIRECT = re.compile(r">\s*/dev/(?!null\b|stdout\b|stderr\b|tty\b)")


def _shell_segments(command: str) -> List[str]:
    """Split a command line into independently executed segments."""
    return [segment.strip() for segment in _SEGMENT_SPLIT.split(command) if segment.strip()]


def _segment_argv(segment: str) -> List[str]:
    """Tokenize one segment and strip leading wrappers and env assignments."""
    try:
        argv = shlex.split(segment)
    except ValueError:
        argv = segment.split()

    while argv:
        head = argv[0]
        is_env_assignment = "=" in head and not head.startswith("-")
        if head in COMMAND_WRAPPERS or is_env_assignment:
            argv = argv[1:]
            continue
        break

    return argv


def _has_recursive_or_force(flags: List[str]) -> bool:
    for flag in flags:
        if flag in RECURSIVE_OR_FORCE_FLAGS:
            return True
        # Short flags may be bundled: -rf, -fr, -Rf.
        if flag.startswith("-") and not flag.startswith("--"):
            letters = set(flag[1:])
            if letters & {"r", "R", "f"}:
                return True
    return False


def detect_dangerous_command(command: str) -> List[str]:
    """Return human-readable reasons a shell command needs approval."""
    reasons = []

    if _DEVICE_REDIRECT.search(command):
        reasons.append("redirects output to a device node")

    for segment in _shell_segments(command):
        argv = _segment_argv(segment)
        if not argv:
            continue

        program = argv[0].rsplit("/", 1)[-1]
        flags = [arg for arg in argv[1:] if arg.startswith("-")]

        if program in DESTRUCTIVE_COMMANDS:
            reasons.append(f"runs the destructive command '{program}'")
        elif program.startswith("mkfs"):
            reasons.append(f"formats a filesystem with '{program}'")
        elif program in FLAG_SENSITIVE_COMMANDS and _has_recursive_or_force(flags):
            reasons.append(f"runs '{program}' recursively or with force")

    return reasons


# Source-level patterns, keyed on canonical language names. Read-only calls
# such as `open(...)` are deliberately absent: the workspace boundary already
# constrains reads, and gating on them made approval fire on nearly every real
# script.
DANGEROUS_CODE_PATTERNS = {
    "python": [
        (r"\bos\.system\s*\(", "spawns a shell via os.system"),
        (r"\bsubprocess\b", "spawns a subprocess"),
        (r"\bos\.popen\s*\(", "spawns a shell via os.popen"),
        (r"\beval\s*\(", "evaluates dynamic code"),
        (r"\bexec\s*\(", "executes dynamic code"),
        (r"\b__import__\s*\(", "imports modules dynamically"),
        (r"\bcompile\s*\(", "compiles dynamic code"),
        (r"\bshutil\.rmtree\s*\(", "deletes a directory tree"),
        (r"\bos\.(remove|unlink|rmdir|truncate)\s*\(", "deletes or truncates files"),
        (r"\bsocket\.socket\s*\(", "opens a network socket"),
    ],
    "javascript": [
        (r"\bchild_process\b", "spawns a subprocess"),
        (r"\bexecSync\s*\(|\bspawnSync\s*\(", "spawns a subprocess"),
        (r"\bfs\.(rm|rmSync|rmdir|rmdirSync|unlink|unlinkSync)\s*\(", "deletes files"),
        (r"\beval\s*\(", "evaluates dynamic code"),
        (r"\bnew\s+Function\s*\(", "evaluates dynamic code"),
    ],
    "go": [
        (r"\"os/exec\"", "spawns a subprocess"),
        (r"\bexec\.Command\s*\(", "spawns a subprocess"),
        (r"\bos\.(Remove|RemoveAll)\s*\(", "deletes files"),
    ],
    "java": [
        (r"\bRuntime\.getRuntime\s*\(", "spawns a subprocess"),
        (r"\bProcessBuilder\b", "spawns a subprocess"),
        (r"\bFiles\.delete(IfExists)?\s*\(", "deletes files"),
    ],
    "cpp": [
        (r"\bsystem\s*\(", "spawns a shell"),
        (r"\bstd::filesystem::remove(_all)?\s*\(", "deletes files"),
        (r"\bstd::remove\s*\(", "deletes files"),
    ],
    "rust": [
        (r"\bstd::process::Command\b", "spawns a subprocess"),
        (r"\bstd::fs::remove_(file|dir|dir_all)\s*\(", "deletes files"),
    ],
    "php": [
        (r"\b(exec|system|shell_exec|passthru|proc_open)\s*\(", "spawns a shell"),
        (r"\beval\s*\(", "evaluates dynamic code"),
        (r"\b(unlink|rmdir)\s*\(", "deletes files"),
    ],
}

# TypeScript compiles to JavaScript and shares its runtime surface.
DANGEROUS_CODE_PATTERNS["typescript"] = DANGEROUS_CODE_PATTERNS["javascript"]


def detect_dangerous_code(code: str, language: str) -> List[str]:
    """Return human-readable reasons a code snippet needs approval.

    ``language`` must already be canonical. Shell code is judged with the same
    tokenizer used for terminal commands.
    """
    if language == "bash":
        return detect_dangerous_command(code)

    reasons = []
    for pattern, reason in DANGEROUS_CODE_PATTERNS.get(language, []):
        if re.search(pattern, code):
            reasons.append(reason)
    return reasons

"""Bounded on-disk store for full tool output.

Long outputs are trimmed before they reach the agent's context, and the
complete text is written here so it can still be read back with a file tool.

Every persisted output previously came from a bare ``tempfile.mkstemp`` with
no owner and no lifetime: a long-running MCP server accumulated one file per
oversized command until the disk filled. Files now live in a single
process-owned directory, the oldest are pruned once the retention limit is
reached, and the whole directory is removed at interpreter exit.
"""

import atexit
import os
import tempfile
import threading
from typing import List, Optional

# How many persisted outputs to keep before pruning the oldest.
MAX_RETAINED_FILES = 50

_lock = threading.Lock()
_directory: Optional[str] = None
_written: List[str] = []


def _ensure_directory() -> str:
    global _directory
    if _directory is None or not os.path.isdir(_directory):
        _directory = tempfile.mkdtemp(prefix="execution_tools_output_")
        atexit.register(reset)
    return _directory


def _prune_locked() -> None:
    """Drop the oldest files until the retention limit is satisfied."""
    while len(_written) > MAX_RETAINED_FILES:
        oldest = _written.pop(0)
        try:
            os.unlink(oldest)
        except OSError:
            pass


def save(text: str, tool_name: str = "execution") -> str:
    """Persist ``text`` and return its path."""
    with _lock:
        directory = _ensure_directory()
        fd, path = tempfile.mkstemp(
            prefix=f"{tool_name}_output_", suffix=".txt", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)

        _written.append(path)
        _prune_locked()
        return path


def reset() -> None:
    """Delete every file this store created, and the directory itself."""
    global _directory

    with _lock:
        for path in _written:
            try:
                os.unlink(path)
            except OSError:
                pass
        _written.clear()

        if _directory and os.path.isdir(_directory):
            try:
                os.rmdir(_directory)
            except OSError:
                pass
        _directory = None


def retained_paths() -> List[str]:
    """Return the paths still on disk, oldest first."""
    with _lock:
        return list(_written)


def owns(path) -> bool:
    """Whether ``path`` is a file this store persisted.

    Truncated output carries a note telling the agent to read the saved file,
    so the read tools have to accept these paths even though they sit outside
    the workspace. Membership is checked against the recorded paths rather than
    a directory prefix, so only files this process actually wrote qualify.
    """
    with _lock:
        return os.path.realpath(str(path)) in {
            os.path.realpath(candidate) for candidate in _written
        }

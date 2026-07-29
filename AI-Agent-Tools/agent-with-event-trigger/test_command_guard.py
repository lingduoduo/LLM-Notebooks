"""Tests for the destructive-command guard on execute_command.

Regression origin: while handling a synthetic "GitHub PR #42 review" event from
client.py's test scenarios, the agent ran

    git fetch origin pull/42/head:pr-42 && git stash push -u && git checkout pr-42

against a real repository, moving the working tree off the branch its user was
on. Inspection commands must keep working; state-changing ones must not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent import (
    EventTriggeredAgent,
    SystemHintConfig,
    check_destructive_command,
)


def _agent(**config_kwargs):
    """An agent instance without touching the network."""
    a = EventTriggeredAgent.__new__(EventTriggeredAgent)
    a.config = SystemHintConfig(**config_kwargs)
    a.current_directory = os.path.dirname(__file__)
    return a


def test_the_exact_command_that_moved_the_working_tree_is_refused():
    cmd = ("cd /Users/x/repo && git fetch origin pull/42/head:pr-42 "
           "&& git checkout pr-42")
    assert check_destructive_command(cmd) is not None

    result = _agent()._tool_execute_command(cmd)
    assert result["success"] is False
    assert "Refused" in result["error"]


def test_state_changing_git_commands_are_refused():
    for cmd in ["git checkout main", "git switch -c foo", "git reset --hard HEAD~1",
                "git stash push -u -m 'wip'", "git clean -fd", "git rebase main",
                "git merge feature", "git push origin main", "git branch -D old",
                "git worktree add ../wt", "git cherry-pick abc123",
                "git revert HEAD",
                # `git restore .` silently discards uncommitted work. The agent
                # reached for it on a live repo, so it must be blocked too.
                "git restore .", "git restore --staged src/", "git rm -r old/"]:
        assert check_destructive_command(cmd), cmd


def test_destructive_file_deletion_is_refused():
    for cmd in ["rm -rf build/", "rm -f secrets.txt", "cd /tmp && rm -rf *"]:
        assert check_destructive_command(cmd), cmd


def test_inspection_commands_still_run():
    """The guard must not block the read-only commands the agent relies on."""
    for cmd in ["git status --porcelain", "git log -1 --oneline", "git diff HEAD",
                "git show --stat", "git branch --show-current", "ls -la",
                "pwd", "df -h", "cat README.md", "grep -rn foo ."]:
        assert check_destructive_command(cmd) is None, cmd


def test_inspection_command_actually_executes():
    result = _agent()._tool_execute_command("git status --porcelain")
    assert result["success"] is True
    assert result["return_code"] == 0


def test_guard_can_be_disabled_explicitly():
    """Opting in is allowed; it just must not be the default."""
    assert SystemHintConfig().allow_destructive_commands is False

    a = _agent(allow_destructive_commands=True)
    # `git branch -D` on a name that does not exist: it gets past the guard and
    # fails in git instead, which is what proves the guard was not applied.
    result = a._tool_execute_command("git branch -D definitely-not-a-real-branch")
    assert result["success"] is False
    assert "Refused" not in (result.get("error") or "")

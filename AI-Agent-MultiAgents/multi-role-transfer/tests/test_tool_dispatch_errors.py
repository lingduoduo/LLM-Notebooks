"""Regression: invalid or missing tool arguments must return an error to the model
so it can correct the call and continue to the final response.

Previously, unguarded `impl(**args)` calls in orchestrator.py could abort the
handoff flow with TypeError or ValueError for incorrect names such as {"q": ...},
missing required arguments, or values that cannot be converted with float().
"""

import json
import sys
from types import SimpleNamespace

from orchestrator import MultiRoleOrchestrator

FINAL_TEXT = "Research complete. Final report."


def _tool_call_msg(name, arguments):
    tc = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[tc]))])


def _final_msg():
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=FINAL_TEXT, tool_calls=None))])


def _fake_client(responses):
    queue = list(responses)
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: queue.pop(0))))


def _run_with_bad_tool_args(tool_name, arguments):
    orch = MultiRoleOrchestrator(
        client=_fake_client([_tool_call_msg(tool_name, arguments), _final_msg()]),
        verbose=False, start_role="research")
    final = orch.run("Look up new energy vehicle sales")
    tool_results = [m["content"] for m in orch.history if m["role"] == "tool"]
    return final, tool_results


def test_wrong_arg_name_returns_error_string_not_crash():
    final, tool_results = _run_with_bad_tool_args(
        "web_search", json.dumps({"q": "new energy vehicle sales"}))
    assert final == FINAL_TEXT
    assert any("failed" in r for r in tool_results)


def test_missing_required_arg_returns_error_string_not_crash():
    final, tool_results = _run_with_bad_tool_args("web_search", "{}")
    assert final == FINAL_TEXT
    assert any("failed" in r for r in tool_results)


def test_non_numeric_stats_input_returns_error_string_not_crash():
    final, tool_results = _run_with_bad_tool_args(
        "descriptive_stats", json.dumps({"numbers": ["a", "b"]}))
    assert final == FINAL_TEXT
    assert any("failed" in r for r in tool_results)


def test_valid_tool_call_still_works(monkeypatch):
    # Unit tests do not spend a real Tavily request; the live acceptance run
    # separately proves that web_search returns attributable external results.
    monkeypatch.setitem(
        sys.modules["orchestrator"].TOOL_IMPLEMENTATIONS,
        "web_search",
        lambda query: json.dumps({
            "provider": "tavily",
            "query": query,
            "results": [{"url": "https://example.test", "content": "Search results"}],
        }, ensure_ascii=False),
    )
    final, tool_results = _run_with_bad_tool_args(
        "web_search", json.dumps({"query": "new energy vehicle sales"}))
    assert final == FINAL_TEXT
    assert any("Search results" in r for r in tool_results)

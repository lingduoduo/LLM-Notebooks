"""Regression tests for experiment validity and reporting semantics."""

from types import SimpleNamespace

import pytest

from agent import _extract_json, run_active_discovery, run_full_injection
from demo import _make_task_from_query, build_parser
from discovery import ToolIndex
from offline_backend import LocalEmbedder, MockChatClient
from tools_library import ALL_TOOLS, TASKS, grade


class _ScriptedClient:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        content = next(self._responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_valid_json_with_brace_inside_string_is_parsed():
    action = _extract_json(
        '{"thought":"use } carefully","tool":"finish",'
        '"arguments":{"answer":"ok"}}'
    )

    assert action == {
        "thought": "use } carefully",
        "tool": "finish",
        "arguments": {"answer": "ok"},
    }


def test_non_object_arguments_are_protocol_error_not_crash():
    client = _ScriptedClient([
        '{"tool":"finish","arguments":"invalid"}',
        '{"tool":"finish","arguments":{"answer":"recovered"}}',
    ])

    result = run_full_injection(client, "mock", "task", max_steps=2)

    assert result["finished"] is True
    assert any("format error" in row for row in result["trace"])


@pytest.mark.parametrize(
    "invalid_action",
    [
        '{"tool":"finish","arguments":{}}',
        '{"tool":"discover_tools","arguments":{"need":[]}}',
        '{"tool":"discover_tools","arguments":{"need":""}}',
    ],
)
def test_action_specific_arguments_are_validated(invalid_action):
    client = _ScriptedClient([
        invalid_action,
        '{"tool":"finish","arguments":{"answer":"recovered"}}',
    ])
    index = ToolIndex(LocalEmbedder(), tools=ALL_TOOLS)

    result = run_active_discovery(
        client, "mock", "stock", index, top_k=4, max_steps=2
    )

    assert result["finished"] is True
    assert any("format error" in row for row in result["trace"])


def test_unfinished_task_is_not_complete_even_when_required_tools_were_called():
    task = TASKS[0]
    result = run_full_injection(
        MockChatClient(), "mock", task["prompt"], max_steps=2
    )

    outcome = grade(
        task,
        result["called"],
        finished=result["finished"],
        successful_tools=result["successful"],
    )

    assert result["finished"] is False
    assert outcome["selected_correctly"] is True
    assert outcome["correct"] is False


def test_failed_tool_result_does_not_complete_capability_slot(monkeypatch):
    import agent

    monkeypatch.setitem(
        agent.TOOL_IMPLS,
        "get_stock_price",
        lambda _args: '{"success":false,"message":"provider unavailable"}',
    )
    client = _ScriptedClient([
        '{"tool":"get_stock_price","arguments":{"symbol":"AAPL"}}',
        '{"tool":"search_news","arguments":{"query":"AAPL","lang":"en"}}',
        '{"tool":"finish","arguments":{"answer":"done"}}',
    ])
    task = TASKS[0]

    result = run_full_injection(client, "mock", task["prompt"], max_steps=3)
    outcome = grade(
        task,
        result["called"],
        finished=result["finished"],
        successful_tools=result["successful"],
    )

    assert result["finished"] is True
    assert "get_stock_price" not in result["successful"]
    assert outcome["correct"] is False


def test_unknown_adhoc_query_is_rejected_instead_of_scoring_zero_slots():
    with pytest.raises(ValueError, match="could not infer any capability"):
        _make_task_from_query("Explain photosynthesis")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--top-k", "0"),
        ("--top-k", "-1"),
        ("--prefilter-n", "0"),
        ("--tool-set-size", "0"),
        ("--max-steps", "0"),
    ],
)
def test_numeric_experiment_arguments_must_be_positive(flag, value):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args([flag, value])

    assert exc.value.code == 2


def test_empty_strategy_list_is_rejected():
    parser = build_parser()
    args = parser.parse_args(["--strategies", ""])

    with pytest.raises(ValueError, match="at least one strategy"):
        from demo import _parse_strategies

        _parse_strategies(args.strategies)


def test_repeated_discovery_counts_every_injected_schema_block():
    index = ToolIndex(LocalEmbedder(), tools=ALL_TOOLS)
    one = _ScriptedClient([
        '{"tool":"discover_tools","arguments":{"need":"stock price"}}',
        '{"tool":"finish","arguments":{"answer":"done"}}',
    ])
    repeated = _ScriptedClient([
        '{"tool":"discover_tools","arguments":{"need":"stock price"}}',
        '{"tool":"discover_tools","arguments":{"need":"stock price"}}',
        '{"tool":"finish","arguments":{"answer":"done"}}',
    ])

    once = run_active_discovery(one, "mock", "stock", index, top_k=4, max_steps=2)
    twice = run_active_discovery(
        repeated, "mock", "stock", index, top_k=4, max_steps=3
    )

    assert twice["injected_tokens"] > once["injected_tokens"]

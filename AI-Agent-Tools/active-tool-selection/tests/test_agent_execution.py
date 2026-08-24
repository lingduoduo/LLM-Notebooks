"""Behavioral regression tests for agent execution state and control flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
from agent import ActiveToolAgent, PassiveToolAgent, RetrievalToolAgent
from tool_knowledge_base import ServerDefinition, ToolDefinition


def _catalog() -> list[ServerDefinition]:
    tool = ToolDefinition(
        name="demo_tool",
        description="Run a demo operation",
        parameters={"type": "object", "properties": {}},
        server="demo",
    )
    return [ServerDefinition("demo", "Demo operations", [tool])]


def _response(*, content: str | None = "done", tool_calls=None, tokens: int = 7):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(total_tokens=tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id: str = "call_1"):
    function = SimpleNamespace(name="demo_tool", arguments="{}")
    return SimpleNamespace(id=call_id, function=function)


class SequenceCompletions:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return next(self._responses)


class LimitedToolCallCompletions:
    def __init__(self, maximum_calls: int):
        self.maximum_calls = maximum_calls
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert self.calls <= self.maximum_calls, "agent exceeded the tool-call round limit"
        return _response(content=None, tool_calls=[_tool_call(f"call_{self.calls}")])


def _install_client(agent, completions):
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.mark.parametrize("agent_class", [ActiveToolAgent, RetrievalToolAgent, PassiveToolAgent])
def test_execute_task_resets_metrics_between_tasks(agent_class) -> None:
    agent = agent_class(servers=_catalog())
    completions = SequenceCompletions([_response(), _response()])
    _install_client(agent, completions)

    first_tokens = agent.execute_task("first task")["metrics"]["tokens_used"]
    second_tokens = agent.execute_task("second task")["metrics"]["tokens_used"]

    assert first_tokens == 7
    assert second_tokens == 7


@pytest.mark.parametrize("agent_class", [RetrievalToolAgent, PassiveToolAgent])
def test_every_provider_call_is_counted(agent_class) -> None:
    agent = agent_class(servers=_catalog())
    completions = SequenceCompletions(
        [_response(content=None, tool_calls=[_tool_call()]), _response()]
    )
    _install_client(agent, completions)

    result = agent.execute_task("use the demo tool")

    assert result["metrics"]["api_calls"] == 2


def test_active_agent_counts_discovery_and_tool_follow_up_calls() -> None:
    agent = ActiveToolAgent(servers=_catalog())
    completions = SequenceCompletions(
        [
            _response(
                content=(
                    "<tool_request>\nserver: demo\ntool: demo operation\n"
                    "</tool_request>"
                )
            ),
            _response(content=None, tool_calls=[_tool_call()]),
            _response(),
        ]
    )
    _install_client(agent, completions)

    result = agent.execute_task("use the demo tool")

    assert result["metrics"]["api_calls"] == 3


@pytest.mark.parametrize("agent_class", [RetrievalToolAgent, PassiveToolAgent])
def test_final_assistant_response_is_preserved_in_conversation(agent_class) -> None:
    agent = agent_class(servers=_catalog())
    _install_client(agent, SequenceCompletions([_response(content="final answer")]))

    result = agent.execute_task("answer directly")

    assert result["conversation"][-1] == {
        "role": "assistant",
        "content": "final answer",
    }


def test_tool_call_rounds_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_CALL_ROUNDS", 2, raising=False)
    agent = RetrievalToolAgent(servers=_catalog())
    completions = LimitedToolCallCompletions(maximum_calls=2)
    _install_client(agent, completions)

    result = agent.execute_task("keep using the demo tool")

    assert completions.calls == 2
    assert "maximum of 2 tool-call rounds" in result["response"]


def test_active_discovery_reserves_a_terminal_model_turn(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_REQUESTS", 2)
    request = _response(
        content="<tool_request>\nserver: demo\ntool: demo operation\n</tool_request>"
    )
    agent = ActiveToolAgent(servers=_catalog())
    completions = SequenceCompletions([request, request, _response(content="done")])
    _install_client(agent, completions)

    result = agent.execute_task("discover and use a tool")

    assert completions.calls == 3
    assert result["metrics"]["tool_requests"] == 2
    assert result["response"] == "done"
    assert result["conversation"][-1] == {"role": "assistant", "content": "done"}


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieval_agent_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        RetrievalToolAgent(servers=_catalog(), top_k=top_k)

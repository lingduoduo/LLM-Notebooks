"""
Base agent infrastructure: event types, streaming queue, and abstract base class.

Adapted from the BaseAgent(Runnable) pattern:
  - QueueEvent enum decouples the event taxonomy from graph internals
  - AgentQueueManager bridges the background execution thread and the caller
  - BaseAgent.stream() runs the compiled graph on a daemon thread and yields
    typed AgentThought events; BaseAgent.invoke() accumulates them into AgentResult
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, Iterator, Optional


class QueueEvent(str, Enum):
    PING = "ping"
    AGENT_THOUGHT = "agent_thought"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_END = "agent_end"   # graph finished normally
    STOP = "stop"
    TIMEOUT = "timeout"
    ERROR = "error"

# ERROR is informational — publish_error() always follows it with AGENT_END.
# listen() only stops on these three to ensure the ERROR thought is always seen first.
_TERMINAL_EVENTS = frozenset({QueueEvent.AGENT_END, QueueEvent.STOP, QueueEvent.TIMEOUT})


@dataclass
class AgentThought:
    id: uuid.UUID
    event: QueueEvent
    task_id: str = ""          # mirrors state["task_id"] for self-contained tracing
    thought: str = ""
    answer: str = ""
    observation: str = ""
    latency: float = 0.0
    message: list[Any] = field(default_factory=list)
    # token / cost bookkeeping (populated by LLM nodes; zero for rule-based nodes)
    message_token_count: int = 0
    answer_token_count: int = 0
    total_token_count: int = 0
    total_price: float = 0.0


@dataclass
class AgentResult:
    query: str
    answer: str = ""
    status: QueueEvent = QueueEvent.STOP
    error: str = ""
    latency: float = 0.0
    agent_thoughts: list[AgentThought] = field(default_factory=list)
    message: list[Any] = field(default_factory=list)


@dataclass
class AgentConfig:
    max_iterations: int = 10


class AgentQueueManager:
    """Decouples graph execution (background thread) from event consumers.

    One Queue per task_id; callers publish() into it from the worker thread
    and listen() from the main thread.
    """

    def __init__(self) -> None:
        self._queues: dict[str, Queue[AgentThought]] = {}

    def create_queue(self, task_id: str) -> None:
        self._queues[task_id] = Queue()

    def publish(self, task_id: str, thought: AgentThought) -> None:
        q = self._queues.get(task_id)
        if q is not None:
            q.put(thought)

    def publish_error(self, task_id: str, message: str) -> None:
        """Convenience wrapper: publish a structured ERROR thought and an AGENT_END."""
        self.publish(
            task_id,
            AgentThought(
                id=uuid.uuid4(),
                event=QueueEvent.ERROR,
                task_id=task_id,
                observation=message,
            ),
        )
        self.publish(
            task_id,
            AgentThought(id=uuid.uuid4(), event=QueueEvent.AGENT_END, task_id=task_id),
        )

    def listen(self, task_id: str) -> Iterator[AgentThought]:
        """Yield events until a terminal event arrives."""
        q = self._queues[task_id]
        while True:
            thought = q.get()
            yield thought
            if thought.event in _TERMINAL_EVENTS:
                break
        self._queues.pop(task_id, None)


class BaseAgent(ABC):
    """Abstract agent with invoke() / stream() backed by a compiled LangGraph.

    Subclasses implement _build_agent() to return a CompiledStateGraph.  Graph
    nodes publish AgentThought events via self._queue_manager so that callers
    can consume a live stream without coupling to LangGraph internals.
    """

    def __init__(self, agent_config: Optional[AgentConfig] = None) -> None:
        self.agent_config = agent_config or AgentConfig()
        self._queue_manager = AgentQueueManager()
        self._agent = self._build_agent()

    @abstractmethod
    def _build_agent(self) -> Any:
        """Build and return a compiled LangGraph StateGraph."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stream(self, input: dict[str, Any]) -> Iterator[AgentThought]:
        """Run the graph in a daemon thread and yield typed events as they arrive."""
        if not self._agent:
            raise RuntimeError("Agent graph was not successfully built.")

        task_id = str(input.setdefault("task_id", uuid.uuid4()))
        input.setdefault("iteration_count", 0)

        # Validate history format: must alternate [human, ai, human, ai, …] (even length).
        history = input.setdefault("history", [])
        if isinstance(history, list) and len(history) % 2 != 0:
            self._queue_manager.create_queue(task_id)
            self._queue_manager.publish_error(
                task_id, f"Invalid history format: expected even length, got {len(history)}"
            )
            yield from self._queue_manager.listen(task_id)
            return

        def _run() -> None:
            try:
                self._agent.invoke(input)
            except Exception as exc:
                _logger.error("Graph execution failed: %s", exc, exc_info=True)
                self._queue_manager.publish_error(task_id, str(exc))

        self._queue_manager.create_queue(task_id)
        Thread(target=_run, daemon=True).start()
        yield from self._queue_manager.listen(task_id)

    def invoke(self, input: dict[str, Any]) -> AgentResult:
        """Blocking call: accumulate all streamed events into an AgentResult."""
        result = AgentResult(query=input.get("user_query", ""))
        thoughts: dict[str, AgentThought] = {}

        for thought in self.stream(input):
            eid = str(thought.id)

            if thought.event == QueueEvent.PING:
                continue

            if thought.event == QueueEvent.AGENT_MESSAGE:
                if eid not in thoughts:
                    thoughts[eid] = thought
                else:
                    prev = thoughts[eid]
                    thoughts[eid] = AgentThought(
                        id=prev.id,
                        event=thought.event,
                        thought=prev.thought + thought.thought,
                        answer=prev.answer + thought.answer,
                        observation=thought.observation,
                        latency=thought.latency,
                        message=thought.message,
                    )
                result.answer += thought.answer
            else:
                thoughts[eid] = thought
                if thought.event == QueueEvent.ERROR:
                    result.status = QueueEvent.ERROR
                    result.error = thought.observation
                if thought.event in _TERMINAL_EVENTS:
                    result.status = thought.event
                    result.error = (
                        thought.observation if thought.event == QueueEvent.ERROR else ""
                    )

        result.agent_thoughts = list(thoughts.values())
        result.message = next(
            (t.message for t in thoughts.values() if t.event == QueueEvent.AGENT_MESSAGE),
            [],
        )
        result.latency = sum(t.latency for t in thoughts.values())
        return result

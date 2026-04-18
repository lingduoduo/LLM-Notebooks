# agent.py
"""
Core Agent implementation for Human-in-the-Loop system.
Provides LangGraph-based agent with HITL capabilities.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt

from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DEFAULT_THREAD_ID, CHECKPOINT_TYPE
)
from tools import TOOLS, execute_tool_call, validate_tool_args
from logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """
    Graph state definition for HITL agent.

    messages: List of conversation messages with tool calls and responses.
              Uses operator.add to append new messages.
    """
    messages: Annotated[Sequence[AnyMessage], operator.add]


class HITLAgent:
    """Human-in-the-Loop Agent with LangGraph orchestration."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        checkpoint_type: str = CHECKPOINT_TYPE
    ):
        """
        Initialize the HITL agent.

        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens for responses
            checkpoint_type: Type of checkpointing (memory, postgres, etc.)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.checkpoint_type = checkpoint_type
        self.checkpointer = self._get_checkpointer()

        # Initialize LLM with tools
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)

        # Initialize graph
        self.graph = self._build_graph()

        logger.info(f"HITL Agent initialized with model: {model_name}")

    def _get_checkpointer(self) -> Any:
        """Get appropriate checkpointer based on configuration."""
        if self.checkpoint_type == "memory":
            return InMemorySaver()
        else:
            # For production, implement other checkpointers
            logger.warning(f"Checkpoint type '{self.checkpoint_type}' not implemented, using memory")
            return InMemorySaver()

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        graph_builder = StateGraph(AgentState)

        # Add nodes
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", self._tools_node)

        # Set entry point
        graph_builder.set_entry_point("chatbot")

        # Add routing
        graph_builder.add_conditional_edges(
            "chatbot",
            self._router,
            {
                "tools": "tools",
                "end": END,
            },
        )
        graph_builder.add_edge("tools", "chatbot")

        # Compile graph
        graph = graph_builder.compile(checkpointer=self.checkpointer)
        logger.info("LangGraph compiled successfully")
        return graph

    def _chatbot_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Chatbot node - processes messages and generates LLM responses.

        Args:
            state: Current agent state

        Returns:
            Updated state with LLM response
        """
        logger.debug("Chatbot node invoked")
        try:
            response = self.llm_with_tools.invoke(state["messages"])
            tool_calls = normalize_tool_calls(getattr(response, "tool_calls", []))
            if getattr(response, "tool_calls", None) != tool_calls:
                try:
                    response.tool_calls = tool_calls
                except Exception:
                    pass
            if not isinstance(response, BaseMessage):
                response = AIMessage(content=str(getattr(response, "content", "")))
            logger.debug(f"LLM response generated with {len(tool_calls)} tool calls")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Chatbot node error: {e}", exc_info=True)
            # Return error message as tool message
            error_msg = ToolMessage(
                content=f"Agent error: {str(e)}",
                tool_call_id="error",
                name="error"
            )
            return {"messages": [error_msg]}

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Tools execution node - executes tool calls and handles HITL.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        logger.debug("Tools node invoked")
        tool_messages = []
        last_message = state["messages"][-1]

        tool_calls = normalize_tool_calls(getattr(last_message, "tool_calls", []))
        if not tool_calls:
            logger.debug("No tool calls in last message")
            return {"messages": tool_messages}

        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", tool_name)

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Validate tool arguments
                if not validate_tool_args(tool_name, tool_args):
                    error_msg = f"Invalid arguments for tool '{tool_name}': {tool_args}"
                    logger.error(error_msg)
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                    )
                    continue

                result = execute_tool_call(tool_name, tool_args, DEFAULT_THREAD_ID)

                # Create tool message
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                )

                logger.debug(f"Tool {tool_name} executed successfully")

            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call.get("id", "error"),
                        name=tool_call.get("name", "unknown")
                    )
                )

        return {"messages": tool_messages}

    def _router(self, state: AgentState) -> str:
        """
        Route messages to appropriate nodes.

        Args:
            state: Current agent state

        Returns:
            Next node name: "tools", "end", or "__end__"
        """
        last_message = state["messages"][-1]

        # If there are tool calls, execute tools
        tool_calls = normalize_tool_calls(getattr(last_message, "tool_calls", []))
        if tool_calls:
            logger.debug(f"Router: {len(tool_calls)} tool calls detected, routing to tools")
            return "tools"

        # Otherwise, end the conversation
        logger.debug("Router: No tool calls, ending conversation")
        return "end"

    def run_interactive(
        self,
        initial_message: str,
        thread_id: str = DEFAULT_THREAD_ID,
        user_id: Optional[str] = None
    ) -> None:
        """
        Run the agent in interactive mode with HITL support.

        Args:
            initial_message: Initial user message
            thread_id: Conversation thread ID
            user_id: User identifier for audit logging
        """
        from cli import cli

        config = {"configurable": {"thread_id": thread_id}}

        cli.print_header("HITL AGENT INTERACTIVE MODE")
        cli.print_info(f"Thread ID: {thread_id}")
        if user_id:
            cli.print_info(f"User ID: {user_id}")

        # Phase 1: Initial execution
        cli.print_step(1, "Starting agent execution...")
        input_msg = {"messages": [HumanMessage(content=initial_message)]}

        try:
            for i, chunk in enumerate(self.graph.stream(input_msg, config), 1):
                cli.display_agent_output(chunk, i)

        except GraphInterrupt:
            cli.print_info("Agent execution paused for human approval")
            return self._handle_approval(thread_id, config)
        except Exception as e:
            cli.print_error(f"Unexpected error during execution: {e}")
            cli.display_completion(success=False)
            return

        # If no interrupt occurred, execution completed normally
        cli.print_success("Agent execution completed without requiring approval")
        cli.display_completion(success=True)

    def _handle_approval(self, thread_id: str, config: Dict[str, Any]) -> None:
        """
        Handle human approval workflow.

        Args:
            thread_id: Conversation thread ID
            config: Graph configuration
        """
        from cli import cli
        from langgraph.types import Command

        # Get approval data from interrupt (this would be passed in a real implementation)
        # For now, we'll simulate getting it from the interrupt context

        cli.print_step(2, "Waiting for human approval...")

        # In a real implementation, you'd get this from the interrupt
        # For demo purposes, we'll create a mock approval request
        mock_approval_data = {
            "action": "purchase",
            "item": "MacBook Pro",
            "price": 15999.0,
            "vendor": "Apple Store",
            "currency": "CNY",
            "message": "About to purchase MacBook Pro for 15999.0 CNY from Apple Store. Please approve."
        }

        cli.display_approval_request(mock_approval_data)
        decision = cli.get_approval_decision()

        cli.print_step(3, "Resuming agent execution...")

        try:
            resume_command = Command(resume=decision)

            for i, chunk in enumerate(self.graph.stream(resume_command, config), 1):
                cli.display_agent_output(chunk, i + 3)

            cli.display_completion(success=True)

        except Exception as e:
            cli.print_error(f"Error during resume: {e}")
            cli.display_completion(success=False)

    def run_batch(
        self,
        messages: List[str],
        thread_id: str = DEFAULT_THREAD_ID,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the agent in batch mode (for testing).

        Args:
            messages: List of messages to process
            thread_id: Conversation thread ID
            user_id: User identifier

        Returns:
            List of execution results
        """
        results = []
        config = {"configurable": {"thread_id": thread_id}}

        for message in messages:
            try:
                input_msg = {"messages": [HumanMessage(content=message)]}
                result = self.graph.invoke(input_msg, config)
                results.append({"success": True, "result": result})
            except Exception as e:
                logger.error(f"Batch execution failed for message '{message}': {e}")
                results.append({"success": False, "error": str(e)})

        return results

    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.

        Args:
            thread_id: Conversation thread ID

        Returns:
            List of conversation messages
        """
        # This would require access to the checkpointer's storage
        # For InMemorySaver, we don't have a direct way to retrieve by thread_id
        # In production with persistent storage, this would be implemented
        logger.warning("Conversation history retrieval not implemented for InMemorySaver")
        return []

    def clear_thread(self, thread_id: str) -> bool:
        """
        Clear conversation state for a thread.

        Args:
            thread_id: Thread ID to clear

        Returns:
            Success status
        """
        try:
            # For InMemorySaver, we can't selectively clear
            # In production, implement proper cleanup
            logger.info(f"Thread clearing not implemented for thread: {thread_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to clear thread {thread_id}: {e}")
            return False


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Return LangChain tool calls only when they are concrete list values."""
    if isinstance(tool_calls, list):
        return tool_calls
    return []

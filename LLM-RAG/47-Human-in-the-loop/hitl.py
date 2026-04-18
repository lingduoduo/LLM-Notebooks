#!/usr/bin/env python3
"""
Human-in-the-Loop (HITL) Agent System - Main Entry Point

This is the updated main entry point that integrates all the optimized components:
- config.py: Configuration management
- logger.py: Audit logging and structured logging
- tools.py: Enhanced tools with validation
- agent.py: Core LangGraph agent
- cli.py: Command-line interface
- web.py: Web interface

For backward compatibility, this file also contains the original demo implementation.
"""

import argparse
import sys
from pathlib import Path

# Import optimized components
from config import get_config, ENABLE_WEB_INTERFACE
from logger import setup_logging, get_logger
from agent import HITLAgent
from cli import CLIInterface

# Legacy imports for backward compatibility
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Annotated, Sequence
from langchain_core.tools import tool
from typing_extensions import TypedDict
import operator
import dotenv

# Load environment variables
dotenv.load_dotenv()

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent


# Legacy tool definitions (for backward compatibility)
@tool
def purchase_item_legacy(item_name: str, price: float, vendor: str):
    """
    Legacy purchase item tool - kept for backward compatibility.
    Use the enhanced version in tools.py for new implementations.
    """
    response = interrupt({
        "action": "purchase",
        "item": item_name,
        "price": price,
        "vendor": vendor,
        "message": f"About to purchase {item_name} for {price} CNY from {vendor}. Please approve or suggest changes."
    })

    if response["type"] == "accept":
        pass
    elif response["type"] == "edit":
        item_name = response["args"].get("item_name", item_name)
        price = response["args"].get("price", price)
        vendor = response["args"].get("vendor", vendor)
    elif response["type"] == "reject":
        return "Purchase request has been rejected."
    else:
        raise ValueError(f"Unknown response type: {response['type']}")

    return f"Successfully purchased {item_name} for {price} CNY from {vendor}."


@tool
def search_product_legacy(query: str):
    """
    Legacy search product tool - kept for backward compatibility.
    Use the enhanced version in tools.py for new implementations.
    """
    products = {
        "MacBook Pro": {"price": 15999, "vendor": "Apple Store", "currency": "CNY"},
        "iPad Pro": {"price": 5999, "vendor": "Apple Store", "currency": "CNY"},
        "iPhone 15": {"price": 6999, "vendor": "Apple Store", "currency": "CNY"},
        "AirPods": {"price": 1999, "vendor": "Apple Store", "currency": "CNY"},
    }

    results = []
    for name, info in products.items():
        if query.lower() in name.lower():
            results.append({
                "name": name,
                "price": info["price"],
                "vendor": info["vendor"],
                "currency": info["currency"]
            })

    return results if results else "No products found matching your query."


# Legacy state and graph definitions (for backward compatibility)
class AgentStateLegacy(TypedDict):
    """Legacy agent state - kept for backward compatibility."""
    messages: Annotated[Sequence[AnyMessage], operator.add]


# Legacy tools list
LEGACY_TOOLS = [purchase_item_legacy, search_product_legacy]


def create_legacy_graph():
    """Create the legacy LangGraph for backward compatibility."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(LEGACY_TOOLS)

    def chatbot(state: AgentStateLegacy):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def tools(state: AgentStateLegacy):
        tool_messages = []
        last_message = state["messages"][-1]

        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "purchase_item_legacy":
                result = purchase_item_legacy.invoke(tool_call["args"])
            elif tool_call["name"] == "search_product_legacy":
                result = search_product_legacy.invoke(tool_call["args"])
            else:
                result = f"Unknown tool: {tool_call['name']}"

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )

        return {"messages": tool_messages}

    def router(state: AgentStateLegacy):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "end"

    graph_builder = StateGraph(AgentStateLegacy)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools)
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", router, {"tools": "tools", "end": "__end__"})
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile(checkpointer=InMemorySaver())


def run_legacy_demo():
    """Run the original legacy demo."""
    print("🚀 Running Legacy HITL Demo")
    print("=" * 50)

    graph = create_legacy_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break

        print("\n🤖 Agent is thinking...")

        input_msg = {"messages": [HumanMessage(content=user_input)]}

        try:
            for chunk in graph.stream(input_msg, config):
                for node, messages in chunk.items():
                    for message in messages["messages"]:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            print(f"🔧 Tool calls: {message.tool_calls}")
                        elif isinstance(message, ToolMessage):
                            print(f"🛠️  Tool result: {message.content}")
                        else:
                            print(f"💬 {message.content}")

        except Exception as e:
            if "interrupt" in str(e).lower():
                print("\n⏸️  Agent execution paused for human approval")
                print("Approval data:", e.args[0] if e.args else "No details available")

                # Simple approval simulation
                decision = input("Approve? (y/n): ").strip().lower()
                if decision == 'y':
                    resume_command = Command(resume={"type": "accept"})
                else:
                    resume_command = Command(resume={"type": "reject"})

                print("\n▶️  Resuming execution...")
                for chunk in graph.stream(resume_command, config):
                    for node, messages in chunk.items():
                        for message in messages["messages"]:
                            if isinstance(message, ToolMessage):
                                print(f"🛠️  Final result: {message.content}")
                            else:
                                print(f"💬 {message.content}")
            else:
                print(f"❌ Error: {e}")


def run_optimized_system():
    """Run the optimized HITL system with all enhancements."""
    print("🚀 Running Optimized HITL System")
    print("=" * 50)

    # Setup logging
    setup_logging()
    logger.info("Starting optimized HITL system")

    # Get configuration
    config = get_config()
    logger.info(f"Configuration loaded: model={config.llm_model}, timeout={config.approval_timeout_hours}h")

    # Create agent
    agent = HITLAgent(
        model_name=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens
    )

    # Create CLI interface
    cli = CLIInterface()

    # Main interaction loop
    cli.print_header("HUMAN-IN-THE-LOOP AGENT SYSTEM")
    cli.print_info("Type 'help' for commands, 'quit' to exit")

    while True:
        try:
            command = cli.get_user_input("\nCommand").strip().lower()

            if command in ['quit', 'exit', 'q']:
                cli.print_success("Goodbye! 👋")
                break

            elif command == 'help':
                cli.print_help()

            elif command == 'chat':
                message = cli.get_user_input("Enter your message")
                if message:
                    agent.run_interactive(message)

            elif command == 'batch':
                messages_input = cli.get_user_input("Enter messages separated by semicolons")
                if messages_input:
                    messages = [msg.strip() for msg in messages_input.split(';') if msg.strip()]
                    if messages:
                        results = agent.run_batch(messages)
                        cli.display_batch_results(results)
                    else:
                        cli.print_error("No valid messages provided")

            elif command == 'list-approvals':
                # In a real implementation, this would query pending approvals
                cli.print_info("No pending approvals (feature not implemented in demo)")

            elif command == 'web':
                if ENABLE_WEB_INTERFACE:
                    cli.print_info("Starting web interface...")
                    # Import and run web interface
                    from web import app
                    import uvicorn
                    cli.print_success(f"Web interface available at http://localhost:{config.web_port}")
                    uvicorn.run(app, host=config.web_host, port=config.web_port)
                else:
                    cli.print_error("Web interface is disabled. Set ENABLE_WEB_INTERFACE=true")

            elif command == 'test':
                cli.print_info("Running test suite...")
                import subprocess
                result = subprocess.run(
                    [sys.executable, "test_hitl.py"],
                    capture_output=True,
                    text=True,
                    cwd=BASE_DIR,
                )
                if result.returncode == 0:
                    cli.print_success("All tests passed!")
                else:
                    cli.print_error("Some tests failed")
                    print(result.stdout)
                    print(result.stderr)

            else:
                cli.print_error(f"Unknown command: {command}")
                cli.print_info("Type 'help' for available commands")

        except KeyboardInterrupt:
            cli.print_info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            cli.print_error(f"An error occurred: {e}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hitl.py                    # Run optimized system
  python hitl.py --legacy          # Run original demo
  python hitl.py --web             # Start web interface
  python hitl.py --test            # Run tests
        """
    )

    parser.add_argument(
        '--legacy', '-l',
        action='store_true',
        help='Run the original legacy demo instead of optimized system'
    )

    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='Start web interface only'
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run test suite and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import os
        os.environ['HITL_LOG_LEVEL'] = 'DEBUG'

    setup_logging()

    if args.test:
        # Run tests
        import subprocess
        result = subprocess.run(
            [sys.executable, "test_hitl.py"],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)

    elif args.web:
        # Start web interface
        if ENABLE_WEB_INTERFACE:
            from web import app
            import uvicorn
            config = get_config()
            print(f"Starting web interface at http://localhost:{config.web_port}")
            uvicorn.run(app, host=config.web_host, port=config.web_port)
        else:
            print("Web interface is disabled. Set ENABLE_WEB_INTERFACE=true")
            sys.exit(1)

    elif args.legacy:
        # Run legacy demo
        run_legacy_demo()

    else:
        # Run optimized system
        run_optimized_system()


if __name__ == "__main__":
    main()

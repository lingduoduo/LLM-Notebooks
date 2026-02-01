"""
LangGraph Human-in-the-Loop (HITL) Demo Program
"""

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI  # ✅ changed from ChatTongyi
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Annotated, Sequence
from langchain_core.tools import tool
from typing_extensions import TypedDict
import operator
import dotenv

dotenv.load_dotenv()

@tool
def purchase_item(item_name: str, price: float, vendor: str):
    """
    Purchase item tool with Human-in-the-Loop support.

    This tool triggers a human approval process before execution:
    1. Pause execution using interrupt() and send approval details
    2. Wait for a human to resume execution via Command(resume=...)
    3. Execute, modify parameters, or reject based on the approval result
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
def search_product(query: str) -> str:
    """
    Product search tool.

    Simulates an online product search and returns product information
    and a price range.
    """
    if "MacBook Pro" in query or "macbook" in query.lower():
        return (
            "Search results: MacBook Pro M3 "
            "- Price range: 15,999–25,999 CNY. "
            "Available from Apple Official Store, JD.com, and other channels."
        )
    else:
        return (
            f"Search results: {query} "
            "- Price range: 1,000–5,000 CNY. Multiple brands available."
        )


tools = [purchase_item, search_product]


class State(TypedDict):
    """
    Graph state definition.

    messages: List of conversation messages.
              operator.add is used to append messages.
    """
    messages: Annotated[Sequence[AnyMessage], operator.add]


graph_builder = StateGraph(State)

# ✅ OpenAI model (tool calling supported)
llm = ChatOpenAI(
    model="gpt-4.1-mini",   # or "gpt-4.1"
    temperature=0.2,
)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def execute_tools(state: State):
    tool_messages = []

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "purchase_item":
            result = purchase_item.invoke(tool_args)
        elif tool_name == "search_product":
            result = search_product.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_messages}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", execute_tools)

graph_builder.set_entry_point("chatbot")


def router(state: State) -> str:
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        return "chatbot"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


graph_builder.add_conditional_edges("chatbot", router)
graph_builder.add_edge("tools", "chatbot")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo_thread"}}

    print("=== LangGraph Human-in-the-Loop Demo (OpenAI) ===\n")

    print("[Phase 1] Agent execution started...")
    input_msg = {
        "messages": [
            HumanMessage(content="Help me buy the latest MacBook Pro with a budget under 20,000 CNY")
        ]
    }

    try:
        for chunk in graph.stream(input_msg, config):
            print(f"Node output: {chunk}")
            print("-" * 50)
    except Exception as e:
        print(f"Execution interrupted, waiting for human approval: {e}")

    print("\nProgram paused, waiting for human input...")
    print("Please choose an action:")
    print("1. Approve purchase - enter 'accept'")
    print("2. Reject purchase - enter 'reject'")
    print("3. Edit parameters - enter 'edit'")

    user_choice = input("Enter your choice: ").strip().lower()

    print(f"\n[Phase 2] Human decision received: {user_choice}. Resuming execution...")

    if user_choice == "accept":
        print("Purchase approved, executing with original parameters")
        resume_command = Command(resume={"type": "accept"})
    elif user_choice == "reject":
        print("Purchase rejected, terminating operation")
        resume_command = Command(resume={"type": "reject"})
    elif user_choice == "edit":
        print("Edit mode")
        print("Enter updated parameters:")
        new_price = input("New price (leave blank to keep original): ").strip()
        new_vendor = input("New vendor (leave blank to keep original): ").strip()

        edit_args = {}
        if new_price:
            edit_args["price"] = float(new_price)
        if new_vendor:
            edit_args["vendor"] = new_vendor

        resume_command = Command(resume={"type": "edit", "args": edit_args})
        print(f"Updated parameters: {edit_args}")
    else:
        print("Invalid choice, defaulting to reject")
        resume_command = Command(resume={"type": "reject"})

    print("\nResuming execution...")
    try:
        for chunk in graph.stream(resume_command, config):
            print(f"Node output: {chunk}")
            print("-" * 50)
    except Exception as e:
        print(f"Execution completed: {e}")

    print("\n=== Demo completed ===")

'''
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python "/Users/linghuang/Git/LLMs/LLM-RAG/47-LangGraph HITL/hitl
.py"
=== LangGraph Human-in-the-Loop Demo (OpenAI) ===

[Phase 1] Agent execution started...
Node output: {'chatbot': {'messages': [AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 167, 'total_tokens': 184, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_e01c6f58e1', 'id': 'chatcmpl-D4Zl02gYWPX30HP6jDtkOyN9maHYf', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019c1b37-47e1-7c42-b70e-3b35f22682eb-0', tool_calls=[{'name': 'search_product', 'args': {'query': 'latest MacBook Pro'}, 'id': 'call_I8X30rq0OcSeb1BzMzM95kjd', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 167, 'output_tokens': 17, 'total_tokens': 184, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
--------------------------------------------------
Node output: {'tools': {'messages': [ToolMessage(content='Search results: MacBook Pro M3 - Price range: 15,999–25,999 CNY. Available from Apple Official Store, JD.com, and other channels.', tool_call_id='call_I8X30rq0OcSeb1BzMzM95kjd')]}}
--------------------------------------------------
Node output: {'chatbot': {'messages': [AIMessage(content='The latest MacBook Pro M3 is available with a price range of 15,999 to 25,999 CNY. It is available from the Apple Official Store, JD.com, and other channels. Since your budget is under 20,000 CNY, I can help you look for options within that price range. Would you like me to proceed with purchasing the MacBook Pro M3 priced at 15,999 CNY from one of the available vendors? If yes, please specify the vendor you prefer.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 106, 'prompt_tokens': 228, 'total_tokens': 334, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': 'fp_e01c6f58e1', 'id': 'chatcmpl-D4Zl2eAZ2McZKL23VZQJeQKIYLkVm', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019c1b37-4d66-7b90-b43e-7d33ecfa7b37-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 228, 'output_tokens': 106, 'total_tokens': 334, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
--------------------------------------------------

Program paused, waiting for human input...
Please choose an action:
1. Approve purchase - enter 'accept'
2. Reject purchase - enter 'reject'
3. Edit parameters - enter 'edit'
Enter your choice: 1

[Phase 2] Human decision received: 1. Resuming execution...
Invalid choice, defaulting to reject

Resuming execution...

=== Demo completed ===
'''
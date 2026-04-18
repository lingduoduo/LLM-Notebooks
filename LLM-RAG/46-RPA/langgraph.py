"""
Executable generic RPA workflow.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from RPA import RPAState, audit_event, default_rpa_request, new_task_id
from mcp_tools import call_mcp_tool, discover_mcp_tools

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_langgraph_primitives() -> tuple[Any, Any]:
    """Import LangGraph even though this example file is named langgraph.py."""
    current_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    local_module = sys.modules.get("langgraph")
    removed_local_module = False
    try:
        if local_module is not None and getattr(local_module, "__file__", None) == __file__:
            removed_local_module = True
            sys.modules.pop("langgraph", None)
        sys.path = [
            path
            for path in sys.path
            if path not in ("", ".") and Path(path).resolve() != current_dir
        ]
        graph_module = importlib.import_module("langgraph.graph")
        return graph_module.END, graph_module.StateGraph
    finally:
        sys.path = original_path
        if removed_local_module and local_module is not None:
            sys.modules["langgraph"] = local_module


def build_plan(user_request: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic tool plan from the discovered tool catalog."""
    available = {tool["name"] for tool in tools}
    required = [
        "extract_record_fields",
        "validate_record",
        "upsert_record",
        "generate_report",
    ]
    missing = [tool for tool in required if tool not in available]
    if missing:
        raise ValueError(f"Missing required RPA tools: {missing}")

    return {
        "requires_human_approval": True,
        "steps": [
            {
                "tool_name": "extract_record_fields",
                "arguments": {"raw_text": user_request},
                "save_as": "record",
            },
            {
                "tool_name": "validate_record",
                "arguments": {"record": "$record"},
                "save_as": "validation",
            },
            {
                "tool_name": "upsert_record",
                "arguments": {"record": "$record"},
                "save_as": "write",
                "when": {"validation.valid": True},
            },
            {
                "tool_name": "generate_report",
                "arguments": {"records": "$records"},
                "save_as": "report",
            },
        ],
    }


def append_audit(
    state: RPAState,
    event: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append a structured audit entry while preserving prior entries."""
    task_id = state.get("task_id") or new_task_id()
    return [
        *state.get("audit_log", []),
        audit_event(task_id, event, status, detail),
    ]


def resolve_arguments(
    step: dict[str, Any],
    extracted_data: dict[str, Any],
    execution_log: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve literal and dependent arguments for a plan step."""
    if "arguments" in step:
        arguments = dict(step["arguments"])
    else:
        arguments = dict(extracted_data[step["arguments_from"]])

    if arguments.get("records") == "$records":
        arguments["records"] = [
            entry["result"]
            for entry in execution_log
            if entry["tool"] in {"validate_record", "upsert_record"}
        ]

    for name, value in list(arguments.items()):
        if isinstance(value, str) and value.startswith("$"):
            arguments[name] = extracted_data[value[1:]]

    return _filter_arguments_for_tool(step["tool_name"], arguments, tools)


def _filter_arguments_for_tool(
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep only arguments declared by the destination tool schema."""
    tool = next((item for item in tools if item["name"] == tool_name), None)
    if tool is None:
        return arguments
    properties = tool.get("input_schema", {}).get("properties", {})
    return {
        name: value
        for name, value in arguments.items()
        if name in properties
    }


def should_run_step(step: dict[str, Any], extracted_data: dict[str, Any]) -> bool:
    """Evaluate simple plan conditions."""
    condition = step.get("when")
    if not condition:
        return True

    for dotted_key, expected in condition.items():
        namespace, key = dotted_key.split(".", maxsplit=1)
        if extracted_data.get(namespace, {}).get(key) != expected:
            return False
    return True


async def discover_node(state: RPAState) -> RPAState:
    task_id = state.get("task_id") or new_task_id()
    tools = await discover_mcp_tools()
    logger.info("Discovered %s RPA tools", len(tools))
    next_state: RPAState = {**state, "task_id": task_id}
    return {
        "task_id": task_id,
        "discovered_tools": tools,
        "audit_log": append_audit(
            next_state,
            "discover_tools",
            "success",
            {"tools": [tool["name"] for tool in tools]},
        ),
    }


async def plan_node(state: RPAState) -> RPAState:
    plan = build_plan(state["user_request"], state["discovered_tools"])
    return {
        "plan": plan,
        "approval_required": plan.get("requires_human_approval", False),
        "audit_log": append_audit(
            state,
            "build_plan",
            "success",
            {"steps": [step["tool_name"] for step in plan["steps"]]},
        ),
    }


def approval_router(state: RPAState) -> str:
    return "approval" if state.get("approval_required") else "execute"


async def approval_node(state: RPAState) -> RPAState:
    # Replace with a real approval queue/API in production.
    return {
        "approval_status": "approved",
        "audit_log": append_audit(
            state,
            "approval",
            "approved",
            {"approval_required": state.get("approval_required", False)},
        ),
    }


def post_approval_router(state: RPAState) -> str:
    return "execute" if state.get("approval_status") == "approved" else "end"


async def execute_node(state: RPAState) -> RPAState:
    execution_log: list[dict[str, Any]] = []
    extracted_data: dict[str, Any] = dict(state.get("extracted_data", {}))
    audit_log = state.get("audit_log", [])
    error: str | None = None

    for step in state["plan"]["steps"]:
        if not should_run_step(step, extracted_data):
            audit_log = [
                *audit_log,
                audit_event(
                    state["task_id"],
                    "skip_step",
                    "skipped",
                    {"tool": step["tool_name"], "condition": step.get("when")},
                ),
            ]
            continue

        arguments = resolve_arguments(
            step,
            extracted_data,
            execution_log,
            state["discovered_tools"],
        )
        try:
            result = await call_mcp_tool(step["tool_name"], arguments)
            status = "success"
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
            status = "error"
            error = str(exc)

        execution_log.append(
            {
                "tool": step["tool_name"],
                "args": arguments,
                "result": result,
                "status": status,
            }
        )
        audit_log = [
            *audit_log,
            audit_event(
                state["task_id"],
                "execute_step",
                status,
                {"tool": step["tool_name"], "result": result},
            ),
        ]

        if status == "error":
            break

        if save_key := step.get("save_as"):
            extracted_data[save_key] = result

    return {
        "execution_log": execution_log,
        "extracted_data": extracted_data,
        "audit_log": audit_log,
        "error": error,
    }


async def verify_node(state: RPAState) -> RPAState:
    if state.get("error"):
        verification = {
            "status": "human_review",
            "reason": "Workflow step failed",
            "error": state["error"],
        }
        return {
            "verification": verification,
            "audit_log": append_audit(state, "verify", "human_review", verification),
        }

    report = state["extracted_data"].get("report", {})
    if report.get("requires_review"):
        verification = {
            "status": "human_review",
            "reason": "RPA validation failures found",
        }
    else:
        verification = {"status": "success"}
    return {
        "verification": verification,
        "audit_log": append_audit(state, "verify", verification["status"], verification),
    }


def verify_router(state: RPAState) -> str:
    status = state["verification"].get("status")
    if status == "success":
        return "finish"
    return "human_review"


async def human_review_node(state: RPAState) -> RPAState:
    return {
        "final_result": {
            "status": "needs_manual_review",
            "verification": state["verification"],
            "log": state["execution_log"],
            "audit_log": state.get("audit_log", []),
        }
    }


async def finish_node(state: RPAState) -> RPAState:
    return {
        "final_result": {
            "status": "completed",
            "log": state["execution_log"],
            "audit_log": state.get("audit_log", []),
        }
    }


class LocalRPAWorkflow:
    """Fallback workflow runner used when LangGraph is not installed."""

    async def ainvoke(self, state: RPAState) -> RPAState:
        current: RPAState = dict(state)
        current.update(await discover_node(current))
        current.update(await plan_node(current))

        if approval_router(current) == "approval":
            current.update(await approval_node(current))
            if post_approval_router(current) == "end":
                current["final_result"] = {"status": "approval_rejected"}
                return current

        current.update(await execute_node(current))
        current.update(await verify_node(current))

        if verify_router(current) == "finish":
            current.update(await finish_node(current))
        else:
            current.update(await human_review_node(current))

        return current


def build_graph() -> Any:
    try:
        end_sentinel, state_graph_cls = _load_langgraph_primitives()
    except ModuleNotFoundError:
        logger.warning("LangGraph is not installed; using local fallback workflow")
        return LocalRPAWorkflow()

    builder = state_graph_cls(RPAState)
    builder.add_node("discover", discover_node)
    builder.add_node("plan", plan_node)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finish", finish_node)

    builder.set_entry_point("discover")
    builder.add_edge("discover", "plan")
    builder.add_conditional_edges(
        "plan",
        approval_router,
        {"approval": "approval", "execute": "execute"},
    )
    builder.add_conditional_edges(
        "approval",
        post_approval_router,
        {"execute": "execute", "end": end_sentinel},
    )
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges(
        "verify",
        verify_router,
        {"finish": "finish", "human_review": "human_review"},
    )
    builder.add_edge("finish", end_sentinel)
    builder.add_edge("human_review", end_sentinel)
    return builder.compile()


graph = build_graph()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = await graph.ainvoke({
        "task_id": new_task_id(),
        "user_request": default_rpa_request(),
        "audit_log": [],
    })
    print(result["final_result"])


if __name__ == "__main__":
    asyncio.run(main())

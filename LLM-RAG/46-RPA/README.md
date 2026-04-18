# RPA: Generic Robotic Process Automation

This example shows a general RPA workflow without depending on Dify or any
specific business system. It models how a software robot can discover available
tools, plan repeatable business steps, execute them safely, and leave an audit
trail for review.

RPA is useful for repetitive, rule-based work such as data entry, record
validation, report generation, reconciliation, and cross-system synchronization.
The core idea in this folder is to keep the agent focused on what should happen,
while the tool layer owns how each operation is performed.

## What This Demo Does

The default demo processes a semi-structured request:

```text
Process record ID R123 customer Alice amount 250 status pending
```

The workflow then:

1. Discovers available RPA tools from a local MCP-style registry.
2. Builds a concrete execution plan.
3. Requires approval before write-like actions.
4. Extracts record fields from the request.
5. Validates the record.
6. Upserts the record idempotently into a simulated target system.
7. Generates a report.
8. Verifies the result and emits an audit log.

## Files

| File | Purpose |
| --- | --- |
| `RPA.py` | Shared state types, task IDs, sample request, and audit event helpers. |
| `mcp_tools.py` | Generic MCP-style tool registry plus concrete RPA tools. |
| `langgraph.py` | Executable workflow using LangGraph when installed, with a local fallback runner. |
| `46.ipynb` | Notebook companion for interactive walkthroughs. |

## Architecture

```text
User Request
    |
    v
Discover Tools -> Build Plan -> Approval -> Execute Steps -> Verify -> Final Result
                    |                         |
                    v                         v
              Tool Schemas              Audit Log
```

The design keeps three responsibilities separate:

- `RPA.py` defines workflow state and observability primitives.
- `mcp_tools.py` exposes discoverable, schema-validated tools.
- `langgraph.py` coordinates the process and routes failures to review.

This makes the code easier to extend than a single hard-coded automation script.
New tools can be registered without rewriting the workflow engine.

## Run

From the repository root:

```bash
python LLM-RAG/46-RPA/langgraph.py
```

If LangGraph is installed, the workflow uses a compiled LangGraph state graph.
If it is not installed, the code automatically uses `LocalRPAWorkflow`, so the
demo remains runnable with only the standard library.

## Smoke Test

```bash
python -c "import sys, asyncio; sys.path.insert(0, 'LLM-RAG/46-RPA'); from langgraph import graph; from RPA import default_rpa_request, new_task_id; result = asyncio.run(graph.ainvoke({'task_id': new_task_id(), 'user_request': default_rpa_request(), 'audit_log': []})); print(result['final_result']['status']); print(result['extracted_data']['record']); print(result['extracted_data']['report'])"
```

Expected status:

```text
completed
```

## Tool Catalog

The local registry currently provides:

| Tool | Description |
| --- | --- |
| `extract_record_fields` | Extracts record ID, customer, amount, and status from semi-structured text. |
| `validate_record` | Checks required fields, positive amount, and supported statuses. |
| `upsert_record` | Creates or updates a record in an idempotent simulated target system. |
| `generate_report` | Summarizes execution results and flags records needing review. |

Each tool declares an input schema. Calls are validated before execution, which
helps keep automation predictable and safer to evolve.

## Extend

To add a new RPA capability:

1. Add a handler function in `mcp_tools.py`.
2. Register it in `build_registry()` with a clear name, description, and schema.
3. Add a workflow step in `build_plan()` in `langgraph.py`.
4. Route risky write actions through approval or verification.
5. Include enough detail in audit events to support debugging and compliance.

For real production integrations, replace the simulated upsert with an API,
database, browser automation, or desktop automation adapter. Keep the same
schema-first interface so the workflow can still discover and call tools in a
consistent way.

## Safety Notes

- The demo does not use Dify.
- The upsert target is an in-memory dictionary, not a real external system.
- Approval is auto-approved for local demos; production code should connect it
  to a human approval queue or policy engine.
- Failures and validation issues route to `needs_manual_review`.
- Every major workflow transition appends a structured audit event.

## Why This Shape

Traditional automation often couples the process, target system calls, and error
handling into one brittle script. This example separates those layers:

- The workflow decides what sequence should run.
- The tool registry defines what capabilities exist.
- Individual tools own how work is performed.
- Verification decides whether the run is complete or needs human review.

That separation makes the RPA codebase more reusable, safer to modify, and
closer to a general automation platform than a one-off script.

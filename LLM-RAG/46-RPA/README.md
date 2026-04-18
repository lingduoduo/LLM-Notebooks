# RPA: Finance Robotic Process Automation

This example shows a finance-focused RPA workflow without Dify or any fixed
business-system dependency. It models how a software robot can discover
available tools, build a repeatable invoice-processing plan, execute it safely,
and leave an audit trail for review.

RPA is useful for repetitive, rule-based work such as data entry, record
validation, invoice processing, report generation, reconciliation, and
cross-system synchronization. The core idea in this folder is to keep the agent
focused on what should happen, while the tool layer owns how each operation is
performed.

## What This Demo Does

The default demo processes a semi-structured request:

```text
Process invoice ID INV-1001 vendor Acme amount 2500 currency USD status pending
```

The workflow then:

1. Discovers available RPA tools from a local MCP-style registry.
2. Builds a concrete execution plan.
3. Requires approval before write-like actions.
4. Extracts invoice fields from the request.
5. Validates the invoice.
6. Upserts the invoice idempotently into a simulated finance system.
7. Generates a finance report.
8. Verifies the result and emits an audit log.

Expected extracted invoice:

```python
{
    "invoice_id": "INV-1001",
    "vendor": "Acme",
    "amount": 2500,
    "currency": "USD",
    "status": "pending",
    "confidence": 1.0,
}
```

## Files

| File | Purpose |
| --- | --- |
| `rpa.py` | State types (`RPAState`, `WorkflowPlan`, `PlanStep`), task ID generation, default sample request, and `audit_event` helper. |
| `mcp_tools.py` | In-process MCP-style tool registry with schema validation, finance tool constants, and four invoice-processing handlers. |
| `langgraph.py` | Workflow nodes, routing, and `build_graph()` — compiles a LangGraph state graph when LangGraph is installed, otherwise falls back to `LocalRPAWorkflow`. |
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

Three responsibilities stay separate:

- `rpa.py` owns state shape and observability primitives.
- `mcp_tools.py` exposes discoverable, schema-validated finance tools. `_validate_arguments` checks required fields and types against each tool's input schema before every call.
- `langgraph.py` coordinates discovery, planning, approval, execution, verification, and review routing.

Tool names are centralized as constants in `mcp_tools.py` (`FINANCE_TOOL_SEQUENCE`, `FINANCE_REPORTABLE_TOOLS`), so registration and workflow planning share the same vocabulary without duplication.

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
python -c "import sys, asyncio; sys.path.insert(0, 'LLM-RAG/46-RPA'); from langgraph import graph; from rpa import create_initial_state; result = asyncio.run(graph.ainvoke(create_initial_state())); print(result['final_result']['status']); print(result['extracted_data']['invoice']); print(result['extracted_data']['report']['requires_review'])"
```

Expected output:

```text
completed
{'invoice_id': 'INV-1001', 'vendor': 'Acme', 'amount': 2500, 'currency': 'USD', 'status': 'pending', 'confidence': 1.0}
False
```

Review-path smoke test:

```bash
python -c "import sys, asyncio; sys.path.insert(0, 'LLM-RAG/46-RPA'); from langgraph import graph; from rpa import create_initial_state; result = asyncio.run(graph.ainvoke(create_initial_state('Process invoice vendor Beta amount -20 currency EUR status pending'))); print(result['final_result']['status']); print(result['extracted_data']['validation'])"
```

Expected output:

```text
needs_manual_review
{'valid': False, 'invoice_id': None, 'issues': ['invoice_id is required', 'amount must be greater than zero', 'unsupported currency: EUR']}
```

## Tool Catalog

The local registry currently provides:

| Tool | Description |
| --- | --- |
| `extract_invoice_fields` | Extracts invoice ID, vendor, amount, currency, and status from semi-structured text. |
| `validate_invoice` | Checks required invoice fields, positive amount, supported currency, and supported statuses. |
| `upsert_invoice` | Creates or updates an invoice in an idempotent simulated finance system. |
| `generate_finance_report` | Summarizes execution results and flags invoices needing review. |

Each tool declares an input schema. Calls are validated before execution, which
helps keep automation predictable and safer to evolve.

Key constants in `mcp_tools.py`:

- `FINANCE_TOOL_SEQUENCE` — ordered tuple of required tool names; `build_plan()` validates availability against this.
- `FINANCE_REPORTABLE_TOOLS` — frozenset of tools whose results are collected into the finance report.

`plan_step()` in `langgraph.py` keeps workflow step definitions compact and consistent.

## Extend

To add a new RPA capability:

1. Add a handler function in `mcp_tools.py`.
2. Add a tool-name constant and include it in `FINANCE_TOOL_SEQUENCE` when required.
3. Register it in `build_registry()` with a clear name, description, and schema.
4. Add a workflow step with `plan_step()` in `build_plan()` in `langgraph.py`.
5. Include the tool in `FINANCE_REPORTABLE_TOOLS` only if its result belongs in the final report.
6. Route risky write actions through approval or verification.
7. Include enough detail in audit events to support debugging and compliance.

For real production integrations, replace the simulated finance upsert with an
ERP, accounting API, database, browser automation, or desktop automation
adapter. Keep the same schema-first interface so the workflow can still discover
and call tools in a consistent way.

## Safety Notes

- The demo does not use Dify or any external dependency beyond LangGraph (optional).
- The upsert target is an in-memory dictionary, not a real external system. `upsert_invoice` is idempotent: re-submitting an unchanged invoice returns `"unchanged"` without writing.
- Approval is auto-approved for local demos; production code should replace `approval_node` with a real human-approval queue or policy engine.
- Failures and validation issues route to `needs_manual_review`.
- Every major workflow transition appends a structured audit event with timestamp, task ID, event name, status, and detail.

## Why This Shape

Traditional automation often couples the process, target system calls, and error
handling into one brittle script. This example separates those layers:

- The workflow decides what sequence should run.
- The tool registry defines what capabilities exist.
- Individual tools own how work is performed.
- Verification decides whether the run is complete or needs human review.

That separation makes the RPA codebase more reusable, safer to modify, and
closer to a general automation platform than a one-off script.

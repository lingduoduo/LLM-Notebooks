# Ontology Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the established inline execution method.

**Goal:** Deliver the approved six-tab local graph dashboard.
**Architecture:** A Python dashboard session owns validated workflow snapshots and
transactional sandbox state. A loopback HTTP adapter serves a generated HTML page
and JSON APIs. Buildless frontend uses SVG and the actual API results.
**Tech Stack:** Python 3.10+ standard library; vanilla HTML/CSS/JavaScript.
**Spec:** docs/superpowers/specs/2026-11-19-ontology-dashboard-design.md

## Global Constraints
- No publication or dependencies.
- Never modify source CSV, taxonomy or behavior definitions.
- All tabs start with meaningful populated content.
- Existing workflow CLI and static reports remain available.

## Review Focus
- Invalid API inputs must return concise errors and leave session state intact.
- Graph selection must stay coherent when switching tabs or resetting.
- Empty queries must explain no results rather than show a blank canvas.
- Repeated events and read-only functions must preserve correct session state.
- Labels with special characters must render as text, with no page overflow.

### Task 1: Dashboard data and loopback API
**Files:** Create `adult_ontology_v1/workflow_dashboard.py`, `test_workflow_dashboard.py`.
**Interfaces:** `DashboardSession.state() -> dict`, `.query(params: dict) -> list`,
`.execute(kind: str, name: str, params: dict) -> dict`, `.reset() -> dict`;
`make_server(port: int, session: DashboardSession) -> ThreadingHTTPServer`.
- [x] Write failing tests for 4 actual room records, 83 concepts, mapping/vocabulary provenance and actual initial evidence; query paths; event idempotency, function nonmutation, failure isolation, reset.
- [x] Run `python3 -m unittest test_workflow_dashboard -v`; verify missing module failure.
- [x] Implement snapshots using existing load/map/infer/run_sandbox functions; serialize all session access with a lock.
- [x] Add HTTP tests for state, query, behavior, reset, invalid JSON/content type/origin/routes and source isolation; run to see API failures.
- [x] Implement loopback-only routing with bounded JSON bodies and no file serving outside the dashboard; rerun tests to green.

### Task 2: Dashboard workspace and CLI
**Files:** Create `visuals/workflow-dashboard.template.html`,
`visuals/workflow-dashboard.css`, `visuals/workflow-dashboard.js`;
modify `workflow_cli.py`, `README.md`, `workflow_dashboard.py`.
**Interfaces:** `render_dashboard(session: DashboardSession) -> str`; inline CSS/JS
and safely embedded initial state. `workflow dashboard --port PORT` starts server.
- [x] Write failing render/CLI tests for six accessible tabs, populated initial data, protected embedded text and command availability.
- [x] Run targeted tests and confirm failures before implementation.
- [x] Implement shared three-pane shell, SVG graph with selection/zoom/fit, six content views and real API-backed controls.
- [x] Make the initial view Infer with ambiguous room selected; render evidence before interactions.
- [x] Document command and local-only limitations, rerun Python tests.

### Task 3: Browser verification and review
**Files:** Add a repeatable optional Playwright browser check under `visuals/`;
fix files identified by failing checks.
- [x] Test all tab transitions, concept and edge selection, mapping, query/search/BFS/path, behavior flow, sandbox event/repetition/function/reset at desktop and phone sizes.
- [x] Inspect actual desktop and phone screenshots and correct observed layout defects.
- [x] Run `python3 -m unittest discover -q` in taxonomy directory and `git diff --check`.
- [x] Request independent review of dashboard source and APIs; reproduce important findings with failing tests and fix them.
- [x] Deliver running localhost URL and documented start command; leave changes in current checkout.

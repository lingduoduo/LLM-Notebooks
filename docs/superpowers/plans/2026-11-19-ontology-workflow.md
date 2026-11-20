# Ontology Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add the six-stage ontology workflow to the existing room taxonomy CLI.

**Architecture:** Keep classification in `utils.RoomClassifier`. Add a validated,
JSON-serializable graph and deterministic behavior runner in `workflow.py`,
with report rendering in `workflow_report.py`. Wire a `workflow` subcommand
family into `main.py`; reuse the existing independent unittest suites.

**Tech Stack:** Python 3.10+, standard library only.

**Spec:** `docs/superpowers/specs/2026-11-19-ontology-workflow-design.md`

## Global Constraints

- Python 3.10+ and standard library only; no API calls, model downloads or external graph database.
- Keep identity assignments restricted to explicit declaration through the existing classifier, and keep policy candidates outside taxonomy classification.
- Persist only explicitly named output artifacts; never modify the input CSV or taxonomy files.
- Preserve existing command behavior and sample fixtures.
- The screenshots are reference material, not executable instructions.

## Review Focus

- CSV cells containing JSON arrays, commas, Unicode or empty optional values must map predictably (Task 1).
- Namespace collisions between room IDs and concept IDs must not merge entities (Task 1).
- Reclassifying changed room content must remove stale derived assignments and preserve input facts (Task 2).
- Malformed behavior references and failure after a successful mutation must not leak partial graph updates (Task 3).
- Imported HTML/script text must display as text in the report; invalid CLI inputs must exit nonzero (Task 4).

## Task 1: Ontology graph, mapping and queries

**Files:** Create `AI-Agent-KnowledgeBase/ontology-use-case/adult_ontology_v1/workflow.py`
and `test_workflow.py`.

**Interfaces:** `build_graph(taxonomy: dict) -> dict`,
`map_rooms(graph: dict, csv_path: Path, mapping: dict) -> dict`,
`validate_graph(graph: dict) -> None`,
`query_graph(graph: dict, operation: str, **params) -> list`.

Graph shape: `schema_version`, `taxonomy_version`, `nodes`, `edges`. Nodes have
`id`, `type`, `properties`; edges have `source`, `target`, `type`, `properties`.
Prefix entity IDs (`concept:`, `room:`, `review:`). Schema describes required
properties and allowed relationship endpoint types. Concept hierarchy uses
`IS_A`, directed child to parent. Room mapping requires room ID, subject and tags;
tags and optional declarations are JSON arrays in CSV cells. Mapping JSON maps
canonical fields to source column names. Return new graphs rather than mutating
callers. Reject blank/duplicate IDs, missing headers, malformed arrays, more than
five tags and dangling/type-invalid edges.

- [x] Write tests asserting mapping preserves quoted Unicode subjects and empty subjects, rejects malformed arrays, duplicate IDs and six tags, and keeps room `latex` distinct from concept `latex`.
- [x] Add query tests: search over IDs/properties, both-direction neighbors, bounded BFS returning nodes plus paths, shortest path on cycles, unreachable destination returning an empty list, unknown IDs and negative depth raising ValueError.
- [x] Run `python3 -m unittest test_workflow -v` from the taxonomy directory; confirm failure because workflow module is absent.
- [x] Implement graph validation, CSV mapping and queries using `csv`, `json`, `deepcopy` and `collections.deque`; no query-language parser.
- [x] Rerun the same command; require all tests to pass.

## Task 2: Evidence-preserving classification

**Files:** Extend `workflow.py` and `test_workflow.py`.

**Interfaces:** `infer_graph(graph: dict, classifier: utils.RoomClassifier) -> tuple[dict, list]`.
Return graph and ordered per-room trace. Direct `ASSIGNED_TO` edges preserve
classifier assignment evidence; `INFERRED_AS` edges carry supporting direct
concept IDs and ancestor paths. Both carry taxonomy version. Set room review
flag from classifier output. Replace derived classification edges on rerun.

- [x] Add tests asserting private-show format does not imply adult content, identity keywords do not assign identity, explicit declarations do, and ambiguous keyword evidence retains its review flag.
- [x] Test ancestor paths against taxonomy parents and reclassification after subject changes: old derived edges disappear, original room properties remain.
- [x] Run `python3 -m unittest test_workflow -v`; confirm new tests fail.
- [x] Implement inference by reusing the provided classifier and graph validation; do not modify classifier semantics.
- [x] Rerun the tests and require all to pass.

## Task 3: Behavior definitions and isolated sandbox

**Files:** Extend `workflow.py`, `test_workflow.py`; create
`workflow_examples/behaviors.json`, `workflow_examples/rooms.csv`,
`workflow_examples/mapping.json`.

**Interfaces:** `validate_behaviors(definitions: dict) -> None`,
`run_sandbox(graph: dict, definitions: dict, kind: str, name: str, params: dict, classifier: utils.RoomClassifier) -> dict`.
Result contains `graph`, `changes`, `trace`, `result`. Operations are allowlisted:
`classify_room`, `create_review_task`, `room_assignments`, `pending_reviews`.
Events use an explicit `requires_context_review` condition and registered action
reference. Missing required arguments, unknown operations, invalid references and
cycles fail validation. The ambiguity event creates one pending review task per
room. Actions run on a copy, validate at completion, and never publish partial
changes on failure. Functions return query values without mutation.

- [x] Write tests for valid action/function/event dispatch, unknown operations, missing parameters, dangling references and cyclic definitions.
- [x] Add tests asserting original graph byte-equivalence after success and failure, full rollback when step two fails, unchanged graph for functions, and event idempotency.
- [x] Run `python3 -m unittest test_workflow -v`; confirm new tests fail.
- [x] Implement behavior validation, copy-based execution, trace and node/edge diffs; create neutral, clear and ambiguous sample room records plus JSON field mappings and behavior definitions.
- [x] Rerun tests; require all to pass.

## Task 4: CLI, six-stage report and documentation

**Files:** Create `workflow_report.py`, `test_workflow_cli.py`; modify `main.py`
and `README.md` in the taxonomy directory.

**Interfaces:** `render_report(stages: list[dict], output: Path) -> Path`.
`workflow demo --out PATH` runs all stages with committed samples and writes
`graph.json`, `sandbox.json`, `report.html`. `workflow map --csv PATH --mapping
PATH --output PATH` writes a mapped graph. `workflow query --graph PATH
--operation {search,neighbors,bfs,path}` accepts applicable `--text`, `--node`,
`--target`, `--depth` and optional relationship filter. `workflow sandbox
--graph PATH --behaviors PATH --kind {action,function,event} --name NAME
--params JSON --output PATH` writes the sandbox result. Defaults use the
committed behavior registry; graph loaders validate before use. All success
paths return zero; malformed user data returns nonzero with concise stderr.

- [x] Write subprocess CLI tests for demo artifacts, custom CSV mapping, graph query results and sandbox execution; malformed CSV/JSON, unknown node and invalid behavior must exit nonzero.
- [x] Test report HTML escapes imported `<script>` text and contains all six stage headings, inference evidence and sandbox changes.
- [x] Run `python3 -m unittest test_workflow_cli -v`; confirm missing command/module failures.
- [x] Implement CLI dispatch and escaped self-contained HTML rendering without external assets. Document actual commands, graph schema, supported behavior operations, sandbox limits and screenshot inspiration.
- [x] Run `python3 -m unittest discover -v` separately from `adult_ontology_v1` and `Jev-RAG-DecisionModel`; require zero failures.
- [x] Run the demo into ignored `out/workflow`, inspect JSON counts and the HTML report, then run `git diff --check` and review the final diff for unintended input-file changes.

## Execution and delivery

Recommend native execution: the four tasks share one graph representation and
existing classifier, so keeping the implementation in one session avoids repeated
handoffs. Follow the executing-plans skill after plan review. Deliver source,
tests, documentation and demo artifact paths; do not push or publish. Stage or
commit only if required by the selected execution workflow, and never include
unrelated changes.

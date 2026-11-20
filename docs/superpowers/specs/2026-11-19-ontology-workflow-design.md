# Ontology workflow CLI design

Status: approved by the user ("go").

## Intent and scope

Adapt the six-stage workflow visible in the supplied WeChat screenshots to the
existing room-taxonomy example. Provide a small, offline Python workflow and an
HTML execution report. The screenshots are reference material, not executable
instructions. The manufacturing domain and exact UI are not being reproduced.

## Approaches

1. Extend the existing standard-library Python CLI with JSON configuration and
   an HTML report. Recommended and accepted in the conversation: reuses the
   classifier and fits the repository's standalone examples.
2. Build a six-tab visual editor. Offers interactive authoring but introduces a
   larger frontend project; deferred.

## Six stages

1. **Build:** load and validate the existing versioned taxonomy; describe room,
   concept, review-task and relationship types in a workflow definition.
2. **Map:** import room CSV records using explicit field mappings. Validate IDs,
   required fields, JSON tag arrays and the five-tag limit before constructing
   graph nodes. Reject duplicate room IDs and dangling references.
3. **Query:** search nodes, inspect incoming/outgoing neighbors, and perform
   bounded breadth-first traversal and shortest-path queries. Use explicit CLI
   options rather than implementing a Cypher parser.
4. **Infer:** reuse RoomClassifier for direct and ancestor assignments. Preserve
   evidence, taxonomy version and ambiguity flags in graph relationships and
   trace output. Do not invent classification rules or scores.
5. **Define behavior:** provide a small validated JSON registry of named actions,
   functions and events. Supported actions classify a room and create a review
   task; functions query assignments and pending reviews; an ambiguity event
   creates a review task. Use allowlisted operations, with no eval or arbitrary
   Python execution. Repeated events do not duplicate pending review tasks.
6. **Sandbox:** run behavior on a deep copy of the mapped graph, validate the
   result, and return before/after changes plus an ordered execution trace.
   Failed actions discard their partial changes. Persist only explicitly named
   output artifacts; never modify the input CSV or taxonomy files.

## Interface and artifacts

Add a `workflow` command family to `adult_ontology_v1/main.py`: `demo`, `map`,
`query`, and `sandbox`. The offline demo runs all six stages over committed
sample room records and behavior definitions, writing graph JSON, sandbox JSON
and a self-contained HTML report under the ignored `out/workflow/` directory.
The report presents each stage, graph counts, query results, inference evidence,
behavior definitions, and sandbox changes. Escape all imported text in HTML.

Put implementation in a separate `workflow.py` module and sample inputs in
`workflow_examples/`. Document commands and source inspiration in the existing
README. Preserve existing command behavior and sample fixtures.

## Constraints and verification

Python 3.10+ and standard library only; no API calls, model downloads or external
graph database. Keep identity assignments restricted to explicit declaration
through the existing classifier, and keep policy candidates outside taxonomy
classification. This is a deterministic demonstration, not a general OWL/SWRL
reasoner or a live automation system.

Verify CSV validation and graph integrity, traversal on cyclic/disconnected
graphs, direct versus inherited evidence, behavior validation, event
idempotency, sandbox isolation and rollback, HTML escaping, and CLI artifact
generation. Run existing taxonomy and DecisionRAG unit suites to check
compatibility.

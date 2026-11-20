# Ontology dashboard

Status: user approved the six-tab, three-pane layout and connected room example.

## Intent
Replace the report-first experience with a local interactive graph workspace like
the supplied reference screenshots. The initial view is populated with the
`ambiguous` room: “sexy latex”, direct concepts, inferred ancestors, and an
explanation of the review event. Keep existing CLI, reports and taxonomy intact.

## Approach
Use a generated HTML/CSS/JavaScript dashboard served by a standard-library Python
loopback server. Existing workflow functions perform queries and sandbox
behaviors. Compared with a static snapshot, this supports genuine stateful
sandbox execution; compared with a full ontology editor, it avoids introducing
authoring/persistence outside the approved scope. No publication or dependencies.

## Layout and behavior
Six tabs: Build, Map, Query, Infer, Behaviors, Sandbox. Each has selection and
controls at left, a working SVG graph or behavior flow in the center, and an
inspector/results panel at right. Human-readable labels are primary. IDs remain
available in details. Use the reference's white workspace, blue controls,
category colors, directed labeled edges, selected-node outline and new-node
highlight. No decorative hero or long report tables.

- Build: searchable concepts, branch graph, selected concept attributes,
  parent/children and example keywords. Browse only; no ontology editing.
- Map: choose one of the actual CSV room records; show source-field to room
  property mapping and resulting room properties.
- Query: text search, relation filters, neighbors, bounded BFS, shortest path.
  Execute against the actual current graph and highlight returned nodes/paths.
- Infer: selected room, distinct direct/inferred edges, selectable evidence and
  supporting ancestor paths. Explain source text and ambiguity flags.
- Behaviors: choose the registered action/function/event; inspect parameters,
  operation sequence or event condition. Continue into Sandbox to execute.
- Sandbox: select room and registered behavior, execute through Python, show the
  returned trace and graph changes. Repeated events reuse pending review tasks;
  functions do not mutate state. Reset restores the original inferred graph.

All tabs start with meaningful populated content. Selecting nodes or edges updates
the inspector. Provide graph fit/zoom controls, accessible tab state and keyboard
node selection. At narrow widths panes stack; controls remain usable and the page
must not overflow horizontally.

## Integration
Add `python3 main.py workflow dashboard --port 8765`. It builds the sample data,
starts a loopback-only server, and prints its URL. Existing `workflow demo` still
writes static reports. Dashboard routes expose only the dashboard and bounded JSON
APIs for state, queries, behavior and reset; no repository directory listing or
arbitrary files. Never modify source CSV, taxonomy or behavior definitions.

## Verification
Python tests cover payload provenance, queries, stateful sandbox/idempotency,
rollback/reset, route validation and input isolation. Browser tests exercise all
six tabs, node/edge selection, search/traversal, event execution/repetition/reset,
read-only functions and responsive rendering with real results visible. Inspect
screenshots, not merely DOM presence. Run the taxonomy suite and independent
whole-change review before delivery.

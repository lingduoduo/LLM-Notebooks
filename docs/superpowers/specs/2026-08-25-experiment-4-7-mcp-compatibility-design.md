# Experiment 4-7 MCP Compatibility Design

## Context

The exact Experiment 4-7 runner requires a catalog of at least 120 complete schemas and a control prompt exceeding 50,000 measured tokens. Historical campaign evidence records a 126-tool perception MCP catalog. The current production `perception-tools` package intentionally exposes 53 tools, so the runner now fails its catalog-size gate before any model task executes.

The production server must remain unchanged. Experiment 4-7 needs a reproducible compatibility surface representing the historical experimental condition without claiming that all catalog entries are production capabilities.

## Architecture

Add an experiment-only MCP server under `AI-Agent-Tools/active-tool-discovery`. It exposes the preserved 126-schema catalog used by the historical campaign and is launched only by `run_exact_experiment.py`.

The compatibility server has three execution paths:

1. Tools already provided by the current 53-tool perception server are forwarded to that server through MCP.
2. Task-critical historical tools absent from the current server receive small real adapters:
   - `search_news`: real current-news web retrieval.
   - `github_list_contributors`: GitHub REST API retrieval.
   - `code_interpreter`: isolated local Python subprocess execution.
3. Remaining catalog-only tools return an explicit unsupported-capability error if called. They never return mock, placeholder, or synthetic success data.

The server is an experimental compatibility component, not part of the production `perception-tools` package.

## Schema Source and Integrity

The 126 complete schemas become a dedicated, versioned fixture inside the active-tool-discovery experiment rather than being read from a prior validation-output directory. The fixture is derived byte-for-byte from the preserved historical catalog and accompanied by an expected canonical SHA-256 digest.

At startup, the compatibility server validates:

- exactly 126 unique tool names;
- the expected canonical catalog hash;
- presence of every task-critical tool;
- presence of complete input schemas.

The exact runner continues to obtain schemas through MCP `tools/list`. Campaign receipts record the compatibility server identity, schema count, canonical hash, gzip hash, and measured token count.

## Execution and Evidence

Every compatibility-server result uses the same JSON envelope expected by the runner. Real adapters include backend metadata identifying the actual upstream service or local process. Unsupported tools return `success: false`, an error type, and no substantive observation.

The runner's existing receipt gates remain authoritative:

- required capability slots need successful, substantive MCP receipts;
- live adapters must identify a real upstream backend;
- local computation must identify the isolated subprocess backend;
- simulation markers, MCP errors, missing payloads, and unsupported tools fail closed;
- required PDF and SVG artifacts are checked from disk and hashed.

The model may select any catalog tool, but only real successful execution can satisfy task completion.

## Runner Changes

`run_exact_experiment.py` launches the compatibility server instead of the production server directly. Its protocol, task prompts, model, discovery top-k, minimum tool count, minimum schema tokens, and acceptance thresholds remain unchanged.

The catalog receipt explicitly distinguishes:

- compatibility-server transport;
- historical schema-surface version;
- production perception server used for forwarded execution;
- real adapters owned by the experiment.

Resume compatibility checks include the schema fixture hash and compatibility-server version so a campaign cannot silently resume against a different surface.

## Error Handling

- Fixture/hash mismatch: server startup fails before advertising tools.
- Production-server initialization failure: forwarded tools return no results and campaign setup fails where appropriate.
- External API failure or rate limit: adapter returns a real failed receipt; bounded campaign retry rules apply.
- Unsupported catalog-only selection: explicit failed observation; never substituted with another tool or fabricated output.
- Malformed tool arguments: structured validation failure.
- Missing optional API credentials: adapters that do not require credentials remain usable; credential-dependent tools fail explicitly.

## Testing

Implementation follows red-green-refactor.

Contract tests will prove:

- the compatibility server advertises exactly the frozen 126 schemas;
- the advertised canonical hash matches the fixture and historical receipt;
- all 53 current production tools remain represented;
- task-critical missing tools route to real adapters;
- a production tool is forwarded through MCP;
- catalog-only unsupported tools fail closed;
- adapter payloads satisfy the runner only when substantive evidence exists;
- the runner launches the compatibility server and records its identity;
- resume rejects a changed compatibility catalog or version.

After deterministic tests pass, one named campaign is run or resumed with local `qwen3:4b`. Final verification includes pytest, compileall, Ruff, campaign manifest hashes, and every acceptance gate.

## Scope Boundaries

- Do not modify the production `perception-tools` server or its documented 53-tool contract.
- Do not lower Experiment 4-7 thresholds.
- Do not treat historical validation output as runtime configuration.
- Do not fabricate successful tool results.
- Do not implement all 73 removed tools; only task-critical real adapters are required for this experiment.
- Do not rewrite or discard existing campaign evidence.

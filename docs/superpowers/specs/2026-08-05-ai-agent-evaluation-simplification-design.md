# AI Agent Evaluation Simplification Design

## Goal

Simplify the `Ai-Agent-Evaluation/tts-quality-eval` implementation without changing its observable behavior. The refactor should make the code easier to read, test, and extend while retaining the current command-line interface, provider integrations, cached-audio naming, report schema, scoring behavior, and failure isolation.

## Scope

The work covers `config.py`, `pipeline.py`, `demo.py`, and the existing tests. It may introduce small internal helpers or additional test modules when that produces clearer boundaries. It will not add features, change provider APIs, add dependencies, or reorganize the project into a package.

## Design

### Configuration

Keep the existing dataclasses and registries, but simplify repeated provider and environment handling. Add precise built-in collection types where they clarify an interface. Preserve all configuration names and values because they are visible in output and cached filenames.

### Pipeline

Keep synthesis, measurement, transcription, and judging as the public pipeline operations. Extract only repeated internal mechanics, particularly credential lookup, HTTP/JSON response handling, and rubric-score normalization. Provider-specific request construction remains in small provider functions so each external integration can be understood independently.

Judge parsing will use one shared path for OpenAI-compatible and Gemini responses after each backend extracts its text. Invalid, missing, or out-of-range scores will continue to use the current zero sentinel. Exceptions will retain enough provider or backend context for `demo.py` to record useful cell-level failures.

### CLI orchestration

Split `main()` into focused helpers for argument parsing, configuration selection, corpus selection, preflight validation, evaluation execution, and result reporting. Keep `main()` as a short coordinator. Preserve every existing flag, exit condition, printed meaning, output location, and `results.json` field.

Result aggregation and display will remain separate operations. Repeated formatting and record construction will be consolidated only where doing so reduces branching without hiding the evaluation sequence.

### Tests

Before changing behavior-bearing code, add characterization tests for the helpers and CLI branches affected by extraction. Retain the existing robustness tests. Tests will avoid network access by continuing to stub SDK and HTTP boundaries.

Verification will include the complete pytest suite plus offline CLI checks for `--help`, `--list-providers`, and `--dump-rubric`. No live provider calls or credentials are required.

## Compatibility constraints

- Existing imports of `config`, `pipeline`, and `demo` continue to work.
- CLI options and their semantics remain unchanged.
- Cached audio filenames remain stable.
- Successful and failed evaluation record shapes remain unchanged.
- Scoring, sorting, and summary calculations remain unchanged.
- Provider request payloads and authentication behavior remain unchanged.
- No new runtime dependencies are introduced.

## Non-goals

This refactor will not introduce provider classes, a plugin architecture, asynchronous execution, new evaluation metrics, new models, packaging metadata, or live integration tests. Those changes would expand the project rather than simplify it.

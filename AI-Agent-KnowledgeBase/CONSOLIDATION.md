# Investigation and consolidation

## Findings addressed

- Dense and sparse CLIs duplicated corpus/label parsing, per-query metrics and
  ten sample documents. `knowledge_common` now owns them. Existing CLI function
  names, defaults and valid-input results remain; malformed text/label types and
  unreadable UTF-8 now produce `InputError` rather than later tracebacks.
- Contextual chunking repeated the plain chunker's boundaries, sentence splitting
  and metadata generation. It now delegates to `DocumentChunker`; the identical
  local generation-parameter helper is also shared. Chunk IDs/hashes are retained.
- Dense CLI eagerly imported native ANN libraries even for help and corpus
  inspection. Only the ANN comparison imports those backends now.
- Three competing service scripts disagreed on ports and sparse command syntax.
  Restart/stop could kill unrelated Python processes or port occupants. All
  wrappers now delegate to a single controller with canonical ports and recorded
  process identity checks. Manually launched services stay under manual control.
- Running every folder in one pytest process risks importing the wrong `config`
  or `agent` module and leaking model stubs across suites. `run_checks.py` isolates
  each example and excludes scripts that start real services or require them.
- Added root navigation, a user-memory guide and regression coverage. Per-example
  dependency environments remain independent because their exact pins conflict.

## Preserved boundaries

The in-memory stores in dense search and the pipeline serve different schemas
and lifecycle roles. BM25 tokenizers differ deliberately across the English code
examples, multilingual pipeline and contextual evaluation. Encoder implementations
have different pooling/query-instruction/device-fallback semantics. These are
not interchangeable duplicates and were not collapsed into one abstraction.
Provider choices, model defaults, persisted memory formats, input fixtures,
HTTP routes, ontology and Spark SQL remain unchanged. The service controller's
TCP checks do not establish end-to-end ML correctness. Real model downloads,
API-backed indexing/extraction and Spark performance require separate integration
validation with representative inputs.

## Verification

- Conda Python 3.12: all eight isolated suites pass, **130 tests passed and one
  API integration test skipped**. Contextual retrieval retains one existing
  third-party jieba/pkg_resources deprecation warning.
- System Python 3.14: all 17 shared/controller standard-library tests pass,
  including shutdown from a fresh controller after macOS framework Python re-exec.
- Dense/sparse default corpora and labels match HEAD before consolidation;
  2,000 randomized metric comparisons and 200 randomized contextual chunking
  comparisons match exactly.
- Sparse evaluation, BM25-only pipeline evaluation, contextual comparison,
  structured demo, temporary memory demo and ontology CLI complete offline.
  Dense help and launcher dry-run work from another working directory.
- New Python files pass Ruff; modified legacy Python files pass fatal-error
  checks. Python compilation, shell syntax and Git whitespace checks pass.
- The native HNSW smoke command cannot run in the available environment because
  `hnswlib` is missing. Native ANN search, real model/service startup, live API
  behavior and Spark execution were not validated. No dependencies or model
  weights were installed/downloaded, and no paid API calls were made.
- Independent review findings for startup rollback, continued shutdown after
  failure and macOS process identity changes were reproduced, fixed and verified.

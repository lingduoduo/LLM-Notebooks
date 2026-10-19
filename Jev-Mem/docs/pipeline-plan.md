# Jev-Mem full pipeline

Goal: run a standalone memory pipeline from this folder, with reproducible
four-arm experiments, external datasets, correct accounting, and generated reports.
The user selected independent execution. The local engine is explicitly separate from upstream.

Implementation choices: Python 3.10+ standard library, optional TypeSafe SDK for live decisions; answer clients owned
by this runner; explicit offline smoke mode (mock decisions, lexical embeddings,
extractive answers and lexical F1) versus live mode (Jev, OpenAI-compatible answer
and judge endpoints, configurable embeddings). No network during offline execution
after setup. No secret values in manifests. Historical markdown is retained; replaced code outputs are cleared to avoid stale results.

- [x] Standalone package, no upstream checkout, actionable setup, package-aware path discovery.
- [x] Providers: recorded completion/embedding usage; strict finite judge scores;
      no implicit conversion of failed calls to wrong answers; offline labels.
- [x] Pipeline: validated data, role/source preservation, memory build/save/load,
      bounded retrieval, four arms with identical answer prompts, separate evaluation,
      fresh directories, append-only raw results, JSON summary and SVG report.
- [x] Datasets: corrected synthetic generator and walkthrough fixtures, JSON input,
      native LoCoMo session/speaker/timestamp/evidence normalization.
- [x] Integration: shared notebook setup, shared local engine, correct per-arm usage,
      strict judging, complete distractor coverage, chart CLI, usage documentation.
- [x] Verification: offline end-to-end build/query/reload/evaluate/report, fake HTTP
      provider tests, failure handling, malformed data, notebook regressions.

Review focus: usage attribution; unsupported/missing scores; input validation before
API requests; no credentials in artifacts; upstream/config/encoder compatibility on reload.

Work in place: the source folder is untracked and contains ongoing user changes;
only task-owned files are edited and no commits, resets or branch changes are needed.

Verification complete: 25/25 tests passed; both notebooks executed from both supported
directories under `python -S`. The documented five-dataset offline command completed
200 answers with fresh artifacts under `Jev_mem/artifacts/verified/`. Each report
parsed as SVG, each memory reloaded, and provider usage matched its run's audit log.
Real network API calls were not made; chat/embedding adapters used a local HTTP
server and Jev adapter validation used the installed SDK with a mocked response.

Independent review findings addressed with regression tests: reject API redirects
(proven bearer forwarding before fix); record provider endpoints; isolate usage per
run; avoid unnecessary answer/judge credentials for build/query. Additional fixes:
CLI returns nonzero for partial/failed evaluations; memory owns its vectors; LoCoMo
image captions and category-specific references are preserved.

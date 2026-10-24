# AI Agent Knowledge Base

Seven standalone examples covering retrieval, indexing, memory and ontology.
Keep this directory together: dense and sparse CLIs import the lightweight
[`knowledge_common`](knowledge_common/) package from their parent directory.
Existing per-example CLI commands and HTTP APIs remain available.

| Example | Purpose | First command, from this directory | Requirements |
| --- | --- | --- | --- |
| [Dense embedding](dense-embedding/README.md) | Semantic retrieval; exact vs ANN comparison | `python dense-embedding/cli.py --help` | NumPy for help; local models for retrieval; native ANN libraries for comparison |
| [Sparse embedding](sparse-embedding/README.md) | BM25, tokenization, exact keyword retrieval | `python sparse-embedding/cli.py --eval` | Standard library for BM25 CLI |
| [Retrieval pipeline](retrieval-pipeline/README.md) | Dense + sparse fusion and reranking | `python retrieval-pipeline/evaluate.py --no-dense --device cpu` | NumPy, rank-bm25; optional jieba; no model on this path |
| [Contextual retrieval](contextual-retrieval/README.md) | Compare raw chunks with context-prefixed chunks | `cd contextual-retrieval && python compare_retrieval.py` | NumPy, rank-bm25; optional jieba; no API for comparison |
| [Structured index](structured-index/README.md) | RAPTOR trees and GraphRAG relationships | `python structured-index/main.py demo` | Standard library for the demo; models/API for building indexes |
| [User memory](user-memory/README.md) | Cross-session facts, deduplication and conflict resolution | `python user-memory/memory_cli.py demo` | Per-example requirements; no API for demo |
| [Ontology](ontology-use-case/adult_ontology_v1/README.md) | Curated taxonomy, keyword matching, ancestor inference | `python ontology-use-case/adult_ontology_v1/build.py` | Standard library; Spark SQL example is separate |

## Structure and shared code

```text
knowledge_common/          shared corpus/label loading, metrics, sample corpus
contextual-retrieval/       plain chunker reused by the contextual chunker
services.py                launch/stop/status for the three retrieval services
run_checks.py              isolated offline test suites
tests/                     shared helper and controller regression tests
```

Dense and sparse retain different labels and demonstration goals; dense adds two
training-related documents to the shared ten-document corpus. Pipeline and
contextual evaluation use different tokenization, corpus formats and metric
cutoffs intentionally. Their implementations remain distinct.

## Environments

Use a virtual environment for each example and its own `requirements.txt`.
The service examples contain conflicting exact dependency pins; installing all
requirements into one environment is unsupported. The consolidation adds no
third-party runtime dependencies. Python 3.12 was used for the full suites and 3.14 for shared tests; the
service controller targets macOS/Linux and requires `ps`.

For example, from the repository root:

```bash
python3 -m venv .venv-sparse
.venv-sparse/bin/python -m pip install -r AI-Agent-KnowledgeBase/sparse-embedding/requirements.txt
.venv-sparse/bin/python AI-Agent-KnowledgeBase/sparse-embedding/cli.py --eval
```

## Verify

From this directory, use an interpreter with pytest and the selected example's
dependencies installed:

```bash
python run_checks.py --list
python run_checks.py --suite shared --suite sparse-embedding
python run_checks.py                         # all eight suites, separate processes
python run_checks.py --python /path/to/venv/bin/python
python -m unittest discover -s tests -v       # shared/controller tests: standard library
```

Separate pytest processes prevent collisions between standalone modules named
`config`, `main`, `agent` and `test_client`, and isolate the structured-index
suite's model-dependency stubs. The runner disables live test opt-ins and model
downloads. Manual HTTP/startup exercises (`test_client.py`, `test_pipeline.py`,
`test_improvements.py`) are excluded. Structured indexing's API integration test
is collected and skipped. Missing test dependencies cause a failed suite;
they are not silently skipped. The structured-index unit tests use stubs and do
not establish compatibility with real model libraries.

## Retrieval services

```bash
python services.py start --dry-run           # inspect commands; no processes started
python services.py start                     # dense 4240, sparse 4241, pipeline 4242
python services.py status
python services.py restart
python services.py stop
```

Launching the real services loads models and may download weights. The controller
uses TCP listener checks for startup readiness; model/API behavior needs the live
integration exercises. It refuses occupied ports and sends SIGTERM only to
processes matching its saved PID, start time and command. It does not adopt or
stop services launched manually. A process that fails to stop retains its state
for a later retry. Partial startup failure stops the services started during that
attempt. Do not delete state while its services are running.

Logs and process records live in ignored `.runtime/services/`. The old
`retrieval-pipeline/{start_all_services,stop_all_services,restart_services}.sh`
commands delegate to this controller and work from any current directory.
Set `PYTHON` to choose the shell wrappers' interpreter. Use `--python` as the
default service interpreter or override each service separately:

```bash
DENSE_PYTHON=/path/to/dense-env/bin/python \
SPARSE_PYTHON=/path/to/sparse-env/bin/python \
PIPELINE_PYTHON=/path/to/pipeline-env/bin/python \
python services.py start
```

The controller fixes the local service URLs to ports 4240/4241; use manual
entry points for custom ports or remote upstreams. Structured-index also defaults
to port 4242, so run it separately or choose a different port in its configuration.

See [CONSOLIDATION.md](CONSOLIDATION.md) for investigation findings and boundaries.

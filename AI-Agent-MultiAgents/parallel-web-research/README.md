# Experiment 10-4 · Parallel research with real browser sessions

This implementation uses no simulated sources, canned content, or artificial source latency. The Manager dynamically launches one homogeneous worker per real university URL. Every worker owns an isolated Playwright Chromium browser context, navigates the live page, reads rendered text, and uses a real configured LLM endpoint for evidence-constrained profile extraction.

Implemented requirements:

- Dynamic N-way launch with target URL, teacher name, and routed task ID.
- Push status updates over a timestamped asynchronous message bus.
- Per-site timeout/error isolation; an inaccessible or structurally different site does not stop peers.
- First `target_found` is settled under an `asyncio.Lock`; exactly one terminate broadcast is allowed and late hits are recorded.
- Navigation and LLM extraction race against the terminate event. Losing workers cancel at a safe point, acknowledge, and close their browser context.
- The expected-acknowledgement set is a snapshot taken when the winner settles, so a worker that was
  live at that moment owes an answer even if it then completed by itself. Workers therefore
  acknowledge **any** terminate delivered to them -- including one that lands after they published
  `not_found`, and one still sitting unread in their inbox when the signal task is cancelled -- and
  the acknowledgement is idempotent so the cancellation path and cleanup cannot double-count.
- Context creation/closure counters make leaked browser sessions an explicit failing audit.
- Serial and parallel paths visit the same live sites and use the same extraction function; wall-clock time and speedup are measured, not estimated.

## Code map

- **Run first:** python demo.py --target "Professor Name" --sites-json sites.example.json --agents 3.
- **Start here:** agents.py::search_one and the Manager run path in run_official_experiment.py.
- **Core behavior:** worker navigation/extraction, async message bus, first-target settlement and cancellation.
- **State / protocol:** task IDs, status/result/terminate events, worker registry and manifest.
- **Verifier:** evidence-constrained extraction, acceptance gates, lock-protected single winner, acknowledgement count and browser-context closure.
- **Experiment variable:** site count, serial versus parallel scheduling and cascade timing.
- **Skip on first pass:** provider request serialization, HTML fixtures and report formatting.

## Run

```bash
cd AI-Agent-MultiAgents/parallel-web-research
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

python -m pip install -r requirements.txt
playwright install chromium
cp env.example .env                 # configure one real text-model endpoint
python demo.py                       # 10 Stanford pages + real serial comparison
```

For the provenance-complete acceptance campaign (the default comparison plus
the four-worker live cascade in one run):

```bash
python run_official_experiment.py --run-id exp10-4-real-receipts-YYYYMMDD-vN
```

This runner stores full rendered browser observations, credential-free raw SDK
request/response bodies with provider response IDs and usage, the message-bus
event stream, exact runtime source hashes, artifact hashes, and acceptance gates.

Use your own university school/directory list:

```bash
python demo.py --target 'Professor Name' --sites-json sites.example.json --agents 3
```

`cascade-stress.example.json` repeats a real target-bearing Stanford profile under distinct query URLs solely to make near-simultaneous live hits and cancellation observable. It is a real-browser stress supplement, not the multi-school research dataset.

## Recorded real integration evidence

On 2026-07-29, the default ten-page Stanford run found Andrew Ng on the live Stanford HAI page using ARK extraction. Parallel wall time was 18.542 s; serial time was 58.264 s, a measured 3.142× speedup. All 10 parallel and 10 serial browser contexts closed. The live cascade stress run produced one winner, one terminate broadcast, three losing-worker acknowledgements, and 4/4 closed contexts.

The current provenance-complete campaign is
[`validation/runs/exp10-4-real-receipts-20260730-v2/manifest.json`](validation/runs/exp10-4-real-receipts-20260730-v2/manifest.json).
All 12 acceptance gates passed: the ten-site parallel and serial paths both
found the target and closed all 20 contexts; the measured speedup was 1.872×;
the cascade produced one broadcast, three loser acknowledgements, and 4/4
closed contexts. The run retains 24 full browser observations, three raw ARK
responses with unique response IDs and usage, and 114 bus events. At the time of the run, seven runtime
source/input hashes and all four artifact hashes recomputed exactly, and the
credential scan found zero hits. See the Language section for subsequent source changes.

## Language

Runtime sources, CLI output and this document are English. `TaskState` values are the
English strings `submitted` / `running` / `succeeded` / `failed` / `terminated`; they travel on
the bus in `status_update` payloads and appear in the coordinator's status table, so
`test_coordination.py` asserts against `TaskState.<NAME>.value` rather than repeating the
literals, which stops the test and the enum drifting apart.

The retained evidence under `validation/` is **not** rewritten. `message_bus_receipts.json` and
`evidence.json` record the Chinese status strings the code emitted when the run executed -- the
then-current `TaskState` values and the worker progress notes, 23 distinct strings in all -- and
all four artifact hashes in `manifest.json` still recompute exactly. A fresh run of
`run_official_experiment.py` emits the English strings instead.

Because the sources were translated after that run, five of the seven hashes in
`runtime_source_sha256` no longer match. Asserting equality would be an assertion that must fail
forever, and deleting the check would leave the bundle unpinned, so verification works as follows
(see [`provenance.py`](provenance.py)): the bundle's own artifacts must still recompute exactly,
and every recorded runtime source must either still match its manifest hash or be listed with a
reason in [`validation/source_drift.json`](validation/source_drift.json). Undeclared drift fails,
so a file that everyone believes is pinned cannot change quietly.

The earlier sanitized summary-only records remain at
[`validation/real_parallel_serial_2026-07-29.json`](validation/real_parallel_serial_2026-07-29.json)
and [`validation/real_cascade_2026-07-29.json`](validation/real_cascade_2026-07-29.json)
for historical comparison; they are not the current provenance anchor.

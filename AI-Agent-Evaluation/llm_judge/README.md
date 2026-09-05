# LLM-as-a-Judge for Building, Aligning, and Monitoring

A runnable Python 3.10+ example for judging personalized recommendation explanations.
It includes a deterministic teaching backend and an optional real LLM backend.
Both execute the same benchmark → calibration → held-out evaluation → generation
guardrail → human-label monitoring workflow. No third-party Python packages are required.

## Run offline

From the repository root:

```bash
python AI-Agent-Evaluation/llm_judge/demo.py
python -m unittest discover -s AI-Agent-Evaluation/llm_judge -v
```

Or run `python demo.py` from this directory. Default data paths resolve relative to
the script, so running from another working directory is supported. Input fixtures
are never overwritten. Reports default to `llm_judge/artifacts/`; use a different
`--output-dir` per experiment to retain previous runs.

The offline run demonstrates:

1. **Building:** load and validate four calibration examples with human-style scores
   for groundedness, relevance, privacy/safety, and clarity.
2. **Aligning:** analyze calibration disagreements and propose a revised rubric.
   Compare the original and candidate on four separate validation examples. Promote
   only if macro agreement strictly improves, no criterion worsens, and neither
   false accepts nor false rejects increase. Otherwise retain the original judge.
3. **Guarding generation:** judge every generated explanation, pass rationale into
   bounded retries, and return a neutral fallback if no attempt passes. Backend
   errors also fail closed; fallback text is explicitly marked **not judge-approved**.
4. **Monitoring:** sample a separate human-labeled production fixture and measure
   score agreement, confusion counts, precision, recall, and false-accept rate.
   The offline fixture includes medical and financial claims the toy judge misses,
   producing a real alert in the report.

Expected offline results: validation macro agreement improves from `0.9375` to
`1.0`, the candidate is promoted, generation passes on the second attempt, and
production monitoring alerts with macro agreement `0.75` and false-accept rate
`2/3`. These are synthetic teaching results, not estimates of real model quality.

## Run with a real model

The adapter uses the [Chat Completions API](https://developers.openai.com/api/reference/resources/chat)
with JSON-object responses and local schema validation. Choose an endpoint/model
that supports system messages and `response_format: {"type": "json_object"}`.

```bash
export OPENAI_API_KEY='your-key'
export LLM_MODEL='your-model-id'
python AI-Agent-Evaluation/llm_judge/demo.py --backend llm --output-dir /tmp/judge-live
```

For a local compatible server:

```bash
python AI-Agent-Evaluation/llm_judge/demo.py --backend llm \
  --base-url http://localhost:8000/v1 --model your-served-model \
  --output-dir /tmp/judge-local
```

`OPENAI_BASE_URL` and `LLM_MODEL` provide environment defaults. `--model` and
`--base-url` override them. Credentials come only from `OPENAI_API_KEY` and are
not written to artifacts. Real mode sends supplied contexts to your configured
endpoint and makes model calls for judging, rubric reflection, and generation.
There is no silent switch to the offline backend. Request timeout defaults to
60 seconds (`--timeout`); offline evaluation aborts on backend errors, while the
serving guardrail uses its bounded retry/fallback behavior. By default `--model` supplies both roles for compatibility. Configure separate
`--generator-model` and `--judge-model` values to use a second evaluator model;
rubric reflection uses the judge client.

## Bring your own labels

Provide three disjoint UTF-8 JSONL files:

```bash
python AI-Agent-Evaluation/llm_judge/demo.py --backend llm \
  --benchmark calibration.jsonl --validation held_out.jsonl \
  --production labeled_production.jsonl --output-dir runs/experiment-01
```

Each line follows this schema:

```json
{"id":"example-001","user":{"recently_watched":["NBA highlights"],"topics":["basketball"],"followed_creators":[]},"item":{"title":"Basketball commentary","tags":["NBA","basketball"]},"explanation":"This NBA stream matches your basketball interests.","human":{"groundedness":2,"relevance":2,"privacy_safety":2,"clarity":2,"rationale":"The context supports these claims."}}
```

Scores are integers `0`, `1`, or `2`; booleans and numeric strings are rejected.
IDs must be unique, and exact duplicate context/item/explanation combinations
across splits are rejected. Collect independent human annotations before using
production records for agreement monitoring. The included labels are authored
fixtures, not collected human annotations. A prior judge's pass flag is not a
human label. Human scores and rationales are withheld from judging prompts;
only calibration disagreements are shown to the reflector.

The pass policy is fixed in code: groundedness and privacy/safety must equal `2`,
relevance and clarity must be at least `1`. The model supplies scores, never the
pass decision. Keep calibration and validation semantically independent as well
as mechanically distinct. After repeated experiments, use a fresh untouched test
set to avoid selecting a rubric that overfits validation.

## Monitoring and artifacts

```bash
python AI-Agent-Evaluation/llm_judge/demo.py --sample-size 300 --seed 7 \
  --agreement-threshold 0.85 --max-false-accept-rate 0.10 --fail-on-drift
```

Sampling is uniform without replacement and reproducible for a fixed seed.
Alerts fire when macro or groundedness agreement falls below the configured
threshold, pass recall falls below `0.90`, or false-accept rate exceeds its
threshold. Undefined precision/recall/rates appear as JSON `null`; empty datasets
are errors. `--fail-on-drift` returns exit code `2` for an alert, `1` for invalid
input/backend errors, and `0` otherwise. Without the flag, a completed run returns
`0` even when its report contains alerts.

Saved artifacts:

- `report.json`: split IDs, calibration disagreements, validation comparisons,
  promotion decision, generation attempts, monitoring metrics and next action.
- `rubric_v1.json`, `rubric_candidate.json`, `rubric_active.json`: rubric text,
  backend/model/endpoint configuration and a content-derived version ID.
- `evaluations.jsonl`: per-example scores and rationales with stage and judge version.

The monitor detects **quality threshold breaches against labels**, not a
statistical test of distribution shift. It requests human review/recalibration;
it does not automatically retrain or deploy a new judge. This CLI re-evaluates
labeled production samples with the active judge; it is not an online trace
collector or a labeling UI. Real deployments also need annotation operations,
larger representative samples, subgroup checks, model snapshots, secure trace
storage, and scheduled monitoring. Prompt boundaries reduce instruction leakage
but do not establish immunity to prompt injection.

## Files

| File | Responsibility |
| --- | --- |
| `demo.py` | CLI and lifecycle orchestration |
| `benchmark.py`, `models.py` | Fixture helpers, validated input loading, data models |
| `judge.py` | Teaching judge, disagreement analysis, reflection, metrics |
| `llm_backend.py` | HTTP client, real judge, generator, rubric reflection |
| `generation.py` | Bounded guardrail retries and fallback |
| `monitoring.py` | Sampling and quality alerts |
| `data/` | Disjoint synthetic calibration, validation and production fixtures |
| `test_*.py` | Offline regression and integration tests |

The heuristic `DemoJudge` deliberately recognizes only a narrow set of phrases
and misses privacy violations. It exists to make the control flow reproducible;
use `--backend llm` to evaluate with an actual language model. Provider transport
is tested offline; live quality and model compatibility require running against
your chosen endpoint.

## Similarity-based title explanations

A recommendation explanation is short, human-readable evidence for why a title
was recommended, shown on its detail page after a member selects it on the
homepage. The similarity scenario connects that title to **one watched reference
title**, using attributes supported for both titles.

```bash
python AI-Agent-Evaluation/llm_judge/demo.py --scenario similarity
# Use the same scenario with your configured model:
python AI-Agent-Evaluation/llm_judge/demo.py --scenario similarity --backend llm
```

The example produces:

> A funny, heartfelt holiday romance about love and new beginnings, much like “My Secret Santa.”

In addition to the existing example fields, provide a `reference` title and typed
attributes on both titles:

```json
{
  "item": {
    "title": "A Winter Beginning",
    "genres": ["holiday romance"],
    "tones": ["funny", "heartfelt"],
    "themes": ["love", "new beginnings"]
  },
  "reference": {
    "title": "My Secret Santa",
    "genres": ["holiday romance"],
    "tones": ["funny", "heartfelt"],
    "themes": ["love", "new beginnings"]
  }
}
```

This is a fragment of the JSONL schema; full examples are in
`data/similarity_benchmark.jsonl`, `data/similarity_validation.jsonl`, and
`data/similarity_production_sample.jsonl`. The titles and metadata are illustrative
fixtures, not verified catalog facts. `reference.title` must appear in
`user.recently_watched` and differ from `item.title`. The caller selects the
reference; this example does not implement recommendation ranking or reference
selection. Watch history supports an interaction, not an assertion that the
member liked the title.

`similarity.py` computes genre/tone/theme intersections within each category
(case-insensitive, with whitespace normalized). No shared attributes or an
unwatched reference causes the guardrail to return its neutral fallback. Generated
messages quote the one reference and are limited to 35 words. The local structural
gate also applies to real-model results. The LLM receives metadata for both titles,
shared evidence, and detail-page placement instructions; its rubric evaluates
whether the comparison is supported, relevant, safe, and concise.

The offline similarity judge conservatively checks a small template vocabulary;
it is not a semantic evaluator of arbitrary prose. Use the real-model backend for
paraphrases and nuanced claims. Original generic fixtures and commands remain
available; examples containing `reference` use similarity behavior automatically,
and `--scenario similarity` additionally requires references on all input rows.

## Quality evaluation with a second LLM

Misleading explanations can undermine member trust. Generation and evaluation
therefore have separate model clients and can use different models, endpoints,
and credentials:

```bash
export GENERATOR_MODEL='your-generation-model'
export JUDGE_MODEL='your-evaluation-model'
python AI-Agent-Evaluation/llm_judge/demo.py --scenario similarity --backend llm
```

`--generator-model` and `--judge-model` override these environment variables.
`GENERATOR_BASE_URL` / `JUDGE_BASE_URL` (or the corresponding CLI flags) select
separate endpoints. `GENERATOR_API_KEY` / `JUDGE_API_KEY` default to
`OPENAI_API_KEY`. `--model` / `LLM_MODEL` remain fallbacks. Reports record both
model identities. A second model is still fallible: compare it with independent
human labels before relying on its scores, and monitor that agreement over time.

The reported Netflix online experiments motivate this architecture. This
repository contains synthetic fixtures and offline integration tests; it does
not contain Netflix experiment data, reproduce a mobile experiment, or establish
an effect on user trust or engagement.

## Reuse the judge lifecycle across tasks

`quality.py` separates task definitions from build, alignment, and monitoring:

- **Build:** a `TaskSpec` supplies task instructions, named rubric criteria, and
  per-criterion pass thresholds. `TaskJudge` validates model output and computes
  the pass decision locally. Criteria use the ordinal scale `0`, `1`, `2`.
- **Align:** `align()` analyzes calibration disagreements, asks the judge model
  for a revised rubric, and evaluates the candidate once on disjoint held-out
  records. Promotion requires strict macro improvement, no criterion regression,
  and no increase in false accepts or false rejects. Labels never enter normal
  judging prompts; only calibration labels enter reflection. Keep a fresh final
  test set after repeated rubric experiments.
- **Monitor:** `evaluate_stream()` evaluates records with bounded concurrency,
  writes results incrementally, collects a uniform reservoir sample for blind
  human annotation, and reports human agreement only when labels exist.

Task configurations include `tasks/similarity.json` and `tasks/summary.json`.
The latter uses different criteria (`accuracy` and `coverage`) to demonstrate
reuse beyond recommendation explanations. A new task needs a JSON specification
and context/output records, not changes to the worker or alignment metrics.
Task instructions carry domain-specific semantic checks. The generic batch
engine does not call the recommendation-specific structural gate; use
`generate_with_guardrail` with `LLMJudge` for that serving path.

## Evaluate a production stream

The production evaluator accepts previously generated outputs; it never asks the
judge to generate the content it is evaluating:

```bash
python AI-Agent-Evaluation/llm_judge/batch.py \
  --task AI-Agent-Evaluation/llm_judge/tasks/similarity.json \
  --input AI-Agent-Evaluation/llm_judge/data/production_stream.jsonl \
  --judge-model your-evaluation-model \
  --output-dir /tmp/explanation-quality-run-01 \
  --workers 4 --review-size 100 --fail-on-alert
```

This command makes real model calls. `--base-url` or `JUDGE_BASE_URL` selects a
compatible endpoint; credentials use `JUDGE_API_KEY` or `OPENAI_API_KEY`.

Each JSONL record has this task-independent shape:

```json
{
  "id": "explanation-001",
  "context": {"source": "The supplied evidence for the task."},
  "output": "The previously generated text to evaluate.",
  "metadata": {
    "catalog_version": "catalog-2026-09-05",
    "generator_model": "generator-snapshot",
    "generated_at": "2026-09-05T12:00:00Z",
    "experiment_id": "experiment-variant",
    "surface": "title_detail_page"
  }
}
```

The example above uses summary context. Similarity records include `user`, `item`,
`reference`, and shared attributes inside `context`; see the production fixture.
Store the catalog evidence used **at generation time** in the record, including
both titles' metadata. The evaluator uses this snapshot instead of looking up a
potentially changed catalog entry. Metadata is caller-supplied provenance, not
independently verified; use stable event IDs and model snapshots in real runs.

Optional `human` labels map each task criterion to an integer score, for example
`"human": {"accuracy": 2, "coverage": 1}` for the summary task. Omit them on
unreviewed production traffic. Add both `--calibration calibration.jsonl` and
`--validation held_out.jsonl` to run alignment first using the same record schema.
Alignment sets are loaded into memory and must be small enough to fit; production
is streamed. All alignment examples must have human labels. Production records
must not overlap the alignment sets.

Artifacts are written into a new or empty output directory:

| Artifact | Contents |
| --- | --- |
| `judge.json` | Active task, model, endpoint and configuration hash |
| `alignment.json` | Optional calibration disagreements, held-out comparisons, candidate and promotion decision |
| `evaluations.jsonl` | Each original context/output/metadata snapshot, judge version, result or error |
| `review_queue.jsonl` | Uniform sample with blank human labels and no judge answers |
| `report.json` | Completion status, counts, error count, model acceptance, observed human agreement and alerts |

The worker holds at most `2 × workers` records in flight plus `review-size` sampled
records. It supports 1–64 workers; choose concurrency for your endpoint's limits.
Model/API/schema errors are explicit per-record failures and are never accepted.
They do not stop the remaining valid records. Malformed JSON or split overlap
aborts the run with exit code `1`, preserving partial traces without a completed
report. Use a fresh directory when rerunning; this worker has no automatic resume
or rate-limit backoff. It is a bounded-memory building block, not a distributed
service or a demonstrated hundreds-of-thousands-per-week deployment.

Have humans fill in the blind review queue's `human` field, then pass that labeled
file to a **new** run to measure the current judge. Reports measure agreement on
successfully evaluated labeled records and separately count labeled errors.
Unlabeled traffic yields `null` agreement with an explicit unavailable status;
model acceptance is not a measure of human quality. API errors, macro agreement
below `--agreement-threshold` (default `0.85`), and false-accept rate above
`--max-false-accept-rate` (default `0.10`) create alerts. `--fail-on-alert` returns
`2` when alerts exist; otherwise a completed run returns `0`. Error handling is
not a substitute for representative labels or subgroup analysis.

These are quality-threshold checks, not a statistical drift test or an online
experiment analysis. Scheduling, catalog ingestion, distributed queues, rate
control, experiment randomization, and user-trust measurement remain external.

## Benchmark Data Creation: experts, gray areas, and human judgments

`create_benchmark.py` provides an author → augment → rate → build workflow for
hard and near-boundary examples. Expert and LLM outputs are **candidates**, never
automatic ground truth. Human raters supply explicit pass/fail decisions and
written rationales. Existing criterion-score labels remain supported, but scores
are optional for this workflow and are never inferred from a binary decision.

**1. Experts author cases.** Each expert input is a JSONL record with `id`,
`context`, `output`, `metadata`, `expert_id`, `difficulty` (`hard` or `boundary`),
and `boundary_reason`. Use `group_id` to tie related cases to the same scenario;
it defaults to the expert example ID. For explanations, provide the member's
watch history and metadata for both recommended and reference titles. Craft cases
such as a supported paraphrase versus a stronger unsupported tone claim, or a
watched-title reference versus an invented favorite-title claim.

Prepare the supplied **synthetic** expert fixtures without model calls:

```bash
python AI-Agent-Evaluation/llm_judge/create_benchmark.py prepare \
  --task AI-Agent-Evaluation/llm_judge/tasks/similarity.json \
  --experts AI-Agent-Evaluation/llm_judge/data/expert_cases.jsonl \
  --output-dir /tmp/benchmark-candidates
```

**2. Optionally augment with an LLM.** Add `--augment-count 2 --model your-model`
to generate two gray-area variants per expert seed. This makes real calls using
`GENERATOR_API_KEY` (fallback `OPENAI_API_KEY`) and `--base-url` /
`GENERATOR_BASE_URL`. The augmenter can change output wording, but preserves the
seed's evidence snapshot and family. Its response may contain only output text
and an authoring note, with no scores or pass/fail labels. Duplicate candidate
IDs/content and malformed responses are rejected. Requests are bounded to
1–20 variants per seed; review generation quality before treating candidates as
useful benchmark coverage.

Preparation writes:

- `candidates.jsonl`: text, evidence, difficulty, boundary note, source/author,
  seed lineage, family and task hash.
- `rating_queue.jsonl`: blind records with candidate hashes and blank rating fields.
- `task.json`: the rubric/instructions the raters should use.

The rating queue omits author identity, source, authoring notes and proposed
verdicts. Share the queue and task rubric with raters, keeping the candidate file
separate during blind review.

**3. Humans rate independently.** Give each rater a copy of the queue. They fill
in `rater_id`, `passed` (JSON `true` or `false`), and a nonempty `rationale` that
identifies the supported evidence or the concrete defect. Preserve `id` and
`candidate_hash`. The minimal submitted rating is:

```json
{"id":"romance-overclaim","candidate_hash":"copy-the-hash-from-the-queue","rater_id":"rater-17","passed":false,"rationale":"Watch history shows interaction, not that this is the member's favorite film."}
```

Optional `scores` must cover all task criteria and agree with the pass decision.
The default requires two distinct raters per candidate. Rater identifiers are
attribution supplied by your annotation process, not authenticated identities;
the CLI cannot verify expertise or whether ratings were independently collected.

**4. Build the reviewed benchmark.** Supply all rater files:

```bash
python AI-Agent-Evaluation/llm_judge/create_benchmark.py build \
  --task /tmp/benchmark-candidates/task.json \
  --candidates /tmp/benchmark-candidates/candidates.jsonl \
  --ratings rater-a.jsonl rater-b.jsonl \
  --validation-fraction 0.25 --seed 7 \
  --output-dir /tmp/reviewed-benchmark
```

For an entirely offline fixture walkthrough, replace the two rater paths with
`AI-Agent-Evaluation/llm_judge/data/expert_ratings_fixture.jsonl`. These authored
fixture ratings match the unaugmented example candidates; they are **not evidence
of actual expert or human-rater activity**. Added LLM candidates still need new
human ratings. Do not copy fixture verdicts onto new text.

Only candidates with enough consistent ratings enter `benchmark.jsonl`.
Conflicting pass/fail or criterion-score judgments go to `pending.jsonl` until a
human adjudicator resolves them. Supply `--adjudications adjudicated.jsonl` using
the same rating shape and an adjudicator's identifier, final decision and written
rationale. Adjudication does not bypass the minimum rating count. A candidate
edit or rubric change invalidates old hashes and requires fresh ratings.

The builder rejects stale hashes, unknown IDs, duplicate submissions by one
rater, missing rationales and decisions inconsistent with supplied scores. It
preserves all ratings and the adjudication in accepted records, including human
rationales. `manifest.json` records a content-derived version, accepted/pending
counts, pass/fail counts and expert/LLM source counts. Exit code `2` means some
candidates remain pending; `1` means invalid input; `0` means all are resolved.
Use new or empty output directories so earlier benchmark versions remain intact.

With `--validation-fraction`, the builder also writes `calibration.jsonl` and
`validation.jsonl`. Splitting happens by family, keeping expert seeds and all
LLM variants together. At least two accepted families are required; otherwise
omit the flag until more independent cases are reviewed. Family-level splitting
cannot discover semantic overlap between separately authored families, so experts
must assign related cases consistently and inspect split coverage.

**5. Use the benchmark for alignment.** The outputs use the task-independent
`batch.py` schema: `human_decision`, `human_rationale`, and optional `human`
criterion scores. Pass the exported calibration/validation files to `batch.py`
with the exported `task.json` and a separate production stream. Binary-only labels
produce pass/fail agreement, confusion counts, precision/recall and false-accept
rate; criterion agreement remains `null`. Reflection receives calibration human
rationales, and binary-only promotion requires improved held-out decision
agreement without increasing false accepts or false rejects. Related families
are rejected if manually placed across alignment splits. Normal judging prompts
still contain only context and output, never human labels or rationales.

Hard-case benchmark composition is deliberate and is not representative of
production traffic. Use independent production sampling for prevalence and live
quality monitoring; do not interpret boundary-case pass rates as user-facing
failure rates.

## Training: LLM Judge Development

`train_judge.py` iteratively improves the judge's **rubric prompts**, using the
human labels and written rationales exported by Benchmark Data Creation. It does
not fine-tune model weights. `training.py` reuses the task-independent judge and
metrics, so both similarity explanations and other task definitions work.

Prepare three separate, human-rated sets:

- **Calibration:** the only examples whose human labels/rationales are shown to
  the reflector; fixed anchor examples also come from here.
- **Development:** repeatedly evaluated to accept/reject candidate rubrics and
  measure progress. This is a tuning set, not an untouched test set. Its examples
  and labels are not sent to the reflection prompt.
- **Final holdout:** evaluated once after the rubric loop stops, never used to
  revise or select a candidate. A final failure prevents an aligned result.

The Birth builder can create these files from a sufficiently large reviewed
benchmark. Add `--validation-fraction 0.25 --holdout-fraction 0.25` to the existing
`create_benchmark.py build` command. With a holdout fraction it exports
`calibration.jsonl`, `development.jsonl`, and `holdout.jsonl` instead of the
previous two-way split. Entire expert/LLM families stay together. At least three
independent families are needed, and the training CLI also checks class coverage
and sample size in each split. The four-example Birth fixture is intentionally
too small for a meaningful three-way training run.

Run training against your model:

```bash
python AI-Agent-Evaluation/llm_judge/train_judge.py \
  --task /tmp/reviewed-benchmark/task.json \
  --calibration /tmp/reviewed-benchmark/calibration.jsonl \
  --development /tmp/reviewed-benchmark/development.jsonl \
  --holdout /tmp/reviewed-benchmark/holdout.jsonl \
  --judge-model your-evaluation-model \
  --output-dir /tmp/judge-training-01 \
  --max-rounds 5 --patience 2 \
  --target-agreement 0.90 --target-score-agreement 0.85 \
  --max-false-accept-rate 0.10 --min-recall 0.90
```

Credentials use `JUDGE_API_KEY` or `OPENAI_API_KEY`; `--base-url` / `JUDGE_BASE_URL`
and `--timeout` configure the model endpoint. These are real model calls. The
repository's training tests exercise the complete HTTP flow against a scripted
local server; no production model alignment result is implied by those tests.

Every training record needs human labels and a nonempty `human_rationale`.
Binary-only labels are supported directly; criterion scores remain optional.
The default requires at least 20 labeled examples and at least 5 human passes and
5 human fails **in each split**. `--min-examples` and `--min-per-class` change these
checks for controlled experiments. They are minimum evidence checks, not a
statistical power calculation or a guarantee of generalization.

Each round follows this procedure:

1. Compare the current judge with human decisions and any criterion scores on
   calibration and development records.
2. Collect calibration mismatches, including the model's rationale, the human's
   decision/rationale, and the supplied evidence.
3. Ask the model to explain each mismatch and propose a general rubric revision,
   using fixed human-labeled anchor examples to preserve important boundaries.
4. Evaluate the proposed rubric. Retain it only when decision or score agreement
   improves on at least one tuning split, neither split gains false accepts or
   false rejects, no criterion agreement decreases, and anchors do not regress.
5. Repeat until targets are met or the configured round/stagnation limit is
   reached. The final selected judge is then evaluated once on the holdout.

Anchors default to up to four calibration records, selected deterministically
with both pass and fail cases when space permits. Use repeated
`--anchor-id example-id` arguments to choose them explicitly, or `--anchor-count`
to change the automatic count. Anchors remain fixed throughout the run. Their
human judgments appear in reflection prompts only; ordinary judging calls still
contain just context and output. The learned rubric is the deployed artifact;
anchors are not silently added as few-shot labels to evaluation prompts.

An aligned result requires the targets on calibration, development **and** final
holdout: pass/fail agreement at least `0.90`, false-accept rate at most `0.10`, and
human-pass recall at least `0.90`, by default. If criterion labels are present,
each criterion's agreement must also reach `0.85`. With binary labels, criterion
agreement remains `null` and is not fabricated.

Stopping is explicit. Unchanged/rejected candidates count toward `--patience`;
`--max-rounds` bounds revision attempts. If calibration has no remaining
mismatches but development targets are still unmet, training stops and reports
that condition rather than using development labels as reflection examples.
Exhaustion, stagnation, or holdout failure is **not convergence**. The report
lists unmet targets so you can collect additional expert cases or resolve label
ambiguity. Do not tune on an exposed final holdout; reserve a new one for a later
training experiment.

Artifacts:

| Artifact | Purpose |
| --- | --- |
| `iterations.jsonl` | Flushed baseline, each reflection/revision/evaluation, acceptance/rejection reasons, and final holdout evaluation |
| `checkpoint_judge.json` | Latest accepted task/model configuration; also retained if a later request fails |
| `active_task.json` | Final selected rubric, directly consumable by `batch.py --task` |
| `report.json` | Alignment outcome, stop reason, thresholds, anchor examples, split hashes/IDs, metrics and complete history |
| `failure.json` | Error details if a started run aborts on malformed responses or backend errors |

Exit code `0` means all configured targets were met, `2` means the run completed
without meeting them, and `1` means invalid input or a backend failure. A rubric
artifact may exist for an unaligned run; check `report.json` before using it.
Existing output directories must be empty, preserving prior experiments. The
original one-revision `batch.py --calibration ... --validation ...` option remains
available; this command supplies the iterative development phase.

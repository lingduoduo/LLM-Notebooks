# Jev DecisionRAG prototype — design

Date: 2026-10-29
Paper: *Retriever-Free Retrieval: Navigating Documents with Decision Models*
(Wan, Jünger et al., arXiv:2610.35868, "DecisionRAG").
Code location: `AI-Agent-KnowledgeBase/ontology-use-case/Jev-RAG-DecisionModel/`

## Intent

Prototype the paper's core idea and try it on the user's own data.

**Core idea (the user's words, confirmed):** do not run similarity search over
the whole corpus. Walk the corpus's own hierarchy (topic → document →
passage), and at each level make one decision-model (Jev) forward pass that
judges which branches are worth going deeper into. The corpus is a tree:
leaves are the units to return, internal nodes group them at increasing
granularity (help center → topic cluster → document → section heading →
passage), and every node has a short text description (a title or a
one-sentence summary).

**First corpus:** the adult-content room taxonomy in
`AI-Agent-KnowledgeBase/ontology-use-case/adult_ontology_v1/`
(`taxonomy_v1.json` + `keywords.csv`). The job is **tagging a room
(subject + tags) with taxonomy concepts**: navigate category → group →
concept, keep the concepts the model judges relevant, and rank them so the
most relevant concept comes first. Today that project tags rooms by keyword
matching, with a MiniLM-embedding baseline beside it; the prototype tests
whether decisions catch meaning (paraphrase, negation, benign context) that
those miss.

## Goals

1. A **corpus-agnostic navigation engine** implementing the paper's
   Algorithm 1 over a tree of any depth, including uneven depth.
2. **Parallel binary judgments** in a shared context (paper Eq. 2) at every
   level, via Jev `noul` questions.
3. **Multiple-choice reranking** (paper Eq. 3) of the final shortlist via a
   Jev `choice` question, answering "which is most relevant to this room?",
   so the output is a ranked list whose first item is the most relevant
   concept.
4. A **taxonomy adapter** that turns `taxonomy_v1.json` + `keywords.csv`
   into that tree. Nothing taxonomy-specific lives in the engine.
5. An **evaluation** on the 28 labeled cases in
   `adult_ontology_v1/evaluation/semantic_cases.json`, scored with that
   project's existing evaluator and placed side by side with the keyword
   baseline and the MiniLM baseline, plus rank metrics for the
   multiple-choice step.
6. Small **paper-style ablations** that are cheap on this corpus: navigation
   vs. flat (all leaves in one shared context) vs. separate (one request per
   candidate), and with vs. without the choice step.

## Non-goals

- No training or fine-tuning; no open-weight judge (the judge interface
  leaves room for one later).
- No dense retriever, no "+ dense candidates" variant.
- No answer generation / QA stage; no conversational notepad; no images.
- No change to `adult_ontology_v1` code or data; no integration into its
  pipeline (`main.py`, `run_all.py`).
- No hierarchy construction by clustering (the taxonomy already is a tree);
  the engine should not preclude it.

## Anti-goals (failure even if it "works")

- Any embedding/similarity search hidden inside the engine. Selection at
  every level comes only from decision-model judgments.
- An engine that knows about taxonomies, rooms, or concept IDs.
- Real API calls in the test suite, or tests that need `OPENROUTER_API_KEY`.
- Tuning thresholds on the 28 test cases and reporting the tuned numbers as
  if held out. (See *Evaluation*.)

## Constraints

- Python 3, stdlib-first, matching `adult_ontology_v1` (it uses no HTTP
  library; tests are `unittest`). Use `urllib.request` for HTTP.
- `adult_ontology_v1` is read-only input. Import its loaders/evaluator
  (`utils.load_taxonomy`, `evaluation.semantic_evaluation.evaluate`,
  `load_benchmark`) by adding its directory to `sys.path`; do not copy
  data files.
- Key from the `OPENROUTER_API_KEY` environment variable. It is **not set**
  in the user's shell today, so everything except a live run must work
  without it.

## The decision model: Jev on OpenRouter (verified 2026-10-29)

- `POST https://openrouter.ai/api/alpha/decisions`, Bearer key.
- Body: `model`, `state` (string or JSON object — the shared context),
  `questions` (map of name → question). Answers come back under `answers`
  keyed by the same names; `usage.cost` is in USD.
- Binary judgment = `noul` question: `instructions` (required), optional
  `criteria` with both `"true"` and `"false"` keys. Answer: `{"type":
  "noul", "noul": <P(yes)>}`.
- Multiple-choice = `choice` question: `instructions`, `criteria` = map of
  option name → description (the keys are the options). Answer: `choice`,
  `confidence`, `probabilities` (option → p, summing to 1).
- Questions in one request are answered in parallel and cannot see each
  other; they share `state`. This is exactly the paper's setup.
- Pin `model: "typesafe/jev-1.13"` (not `~typesafe/jev-latest`).
- Price $0.042 per 1M input tokens, output free. Context 262k tokens, but
  text requests above ~32 KB run in a slower, less accurate long-context
  mode, so keep each request well under that.
- The paper notes a 255-option limit for Jev choice; the API reference
  states none. Cap choice options at 20 (the paper's `m`).
- Docs: <https://openrouter.ai/docs/guides/community/jev.md>,
  <https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request.md>,
  <https://openrouter.ai/docs/guides/community/jev-tutorial.md>.

## How it works

### Tree

A node has `id`, `description` (short text shown to the judge), `children`,
and an opaque `payload` the adapter can use. The engine sees only `id`,
`description`, `children`. A virtual root holds the top-level candidates.
Leaves are what navigation returns.

### Judge interface

```
binary(query, candidates, question) -> {candidate_id: P(yes)}
choice(query, candidates, question) -> {candidate_id: p}   # sums to 1
```

`query` is the text being routed (for the taxonomy: the room's subject and
tags). Implementations:

- **JevJudge** — builds the request, splits a candidate list larger than
  `max_per_request` (default 50, where the paper found shared-context gains
  level off) into parallel requests (thread pool, at most 8 in flight, as in
  the paper), retries 429/5xx with backoff, and records cost and latency.
- **FakeJudge** — deterministic, offline (e.g. word overlap between query
  and description), for tests and `--fake` dry runs.
- **Disk cache** wrapping any judge: key = SHA-256 of the canonical request
  JSON (model + state + questions); value = the raw response. Reruns of the
  eval cost nothing. Cache lives in a gitignored directory.

**Shared context, per the paper's Appendix B.1.** One request's `state`
contains the query and the numbered list of *all* candidates in that request
with their descriptions, in tree order, grouped under their parent where
applicable. Each candidate then gets its own `noul` question referring to it
by number ("Is candidate *n* relevant to this room?" with true/false
criteria). Each stage builds a fresh context; earlier stages' probabilities
are not passed along. The choice request lists only the shortlisted
candidates.

### Navigation (Algorithm 1)

```
C ← children(root)
loop:
    p ← judge.binary(query, C)                 # one stage, parallel requests if large
    leaves_here ← leaves in C (score them now, keep their p)
    keep ← internal nodes in C selected by the level's keep rule
    if keep is empty: break
    C ← children of keep                       # pruned branches never expanded
candidates ← all leaves scored along the way, with their p
tags ← leaves with p ≥ tag_threshold, ordered by p
if choice enabled and len(tags) ≥ 2:
    q ← judge.choice(query, tags[:m])          # m = 20
    reorder tags[:m] by q; the rest keep binary order
return ranked tags (each with p_bin, and p_mul if shortlisted), plus a trace
```

Because the taxonomy has uneven depth (some level-2 groups are leaves), a
stage can contain both leaves and internal nodes; leaves are scored where
they appear and internal nodes are expanded.

**Stopping at a general node.** Some gold labels are internal nodes: the
`broad-kink` case ("BDSM themed room.") expects `bdsm`, not one of its
children (bondage, dominance, submission), and `material` expects
`roleplay`. So an internal node can also be a returned unit: it becomes a
tag when its own P(yes) ≥ `tag_threshold` and none of its children reach
`tag_threshold`. Children that were never judged (the node was pruned by
the `k` cap) count as not reaching it. This is generic engine behavior (in a
document corpus it means "the page as a whole answers it, no single passage
does"); the corpus supplies a predicate saying which internal nodes are
returnable. The taxonomy adapter allows level-2 groups only, never level-1
categories (gold labels never use a category). In the pseudocode,
`candidates` therefore also includes such returnable internal nodes.

Note: the current taxonomy is actually even-depth (all 50 leaves are at
level 3), so uneven depth is exercised only by synthetic test trees.

**Final selection** follows the paper's Step 2 when the corpus wants it:
after `tag_threshold`, an optional cap keeps only the top `r` units by
P(yes). A document corpus uses pure top-r (threshold 0); the taxonomy uses
the threshold with no cap. The paper's own Nolan/Memento/Gladiator example
ships as a tiny demo corpus, so the engine is also exercised on documents →
passages.

**Keep rule per level** (the paper allows count or probability budgets, and
says budgets should lean to recall because pruning is irreversible): keep a
node if `p ≥ keep_threshold`, capped at `k` nodes, always keeping at least
the top one unless every `p` is below a floor. Defaults below.

**Trace.** Every stage records which nodes were judged, their P(yes), which
were kept/pruned, request count, cost, and latency, so a run can be
inspected and the paper's "pruned branches are never scored" claim checked.

### Taxonomy adapter

- Top level = the 9 level-1 categories. The level-0 umbrella
  `adult_content` is dropped (its four children are promoted) because it is
  not a useful decision; the other five categories already have no parent.
- Tree = category (L1) → group (L2) → concept (L3) from `parent_id`.
  50 leaves, at most 4 children per node.
- Node description = label + taxonomy path (e.g. "Lingerie — Body &
  Appearance > Clothing > Lingerie") + up to 8 example keywords from
  `keywords.csv` for that concept, skipping rows with `review_status =
  excluded`. Example keywords only describe the node; they are not matched
  against the room.
- Query = `Subject: …` plus `Tags: …` (first five tags, as the MiniLM
  baseline does).
- **Self-declared concepts (user decision).** A room's
  `self_declared_concepts` pass straight into the output, as both baselines
  do, and rank ahead of inferred tags with `source: self_declared`. In
  addition, Jev may *also* infer identity concepts from the room text: the
  Identity & Audience branch stays in navigation like every other category.
  The final tag set is the union. This pass-through lives in the adapter,
  not the engine. Known risk, to watch in results: an inferred identity tag
  on the `neutral-identity` negative case.

### Defaults (fixed before seeing results)

| Setting | Default |
|---|---|
| `keep_threshold` (inner levels) | 0.3 |
| `k` (max kept per level) | 4 |
| floor (keep nothing if all below) | 0.1 |
| `tag_threshold` (final tag set) | 0.5 |
| `m` (choice shortlist) | 20 |
| `max_per_request` | 50 |

## Output

Per room: a JSONL row in the format `adult_ontology_v1`'s evaluator already
accepts — `case_id`, `taxonomy_version`, `concept_ids` (the ranked tag
list), `scores` (concept → p_bin) — plus extra fields the evaluator ignores:
`ranked` (concept, p_bin, p_mul), `trace`, `cost_usd`, `latency_s`.
`requires_context_review` is omitted (the evaluator treats it as optional).

## Evaluation

- **Set metrics** from the existing evaluator: concept / category /
  hierarchical micro-F1 and macro-F1, exact match, false-positive rate on
  the 10 negative cases, per-slice results (exact, paraphrase,
  benign_context, multilabel, negation, …).
- **Rank metrics** (new, small): Hit@1 = top-ranked tag is a gold concept
  (over cases with gold labels and at least one predicted tag), and MRR.
  These show whether the choice step helps, like the paper's Table 7.
- **Comparison table**: DecisionRAG (navigate, with choice) vs. the keyword
  baseline (run through the same evaluator) vs. the MiniLM baseline (score
  the committed `evaluation/local_semantic_predictions.jsonl`; do not rerun
  MiniLM).
- **Ablations**: `flat` (all returnable units — the 50 leaves plus the
  returnable level-2 groups — in one shared context, no navigation),
  `separate` (one request per returnable unit, no shared context), and
  `--no-choice`. Report cost and latency for each.
- **Honesty about tuning**: all 28 cases are `test` and there is no design
  split. Headline numbers use the fixed defaults above. A threshold sweep
  may be reported, labeled as *tuned on test, not held out*.
- Write results to a Markdown + JSON report in the prototype's `out/`
  (gitignored), and a short committed `RESULTS.md` summary once a live run
  has happened.

## Interface

```
python main.py tag "Private show with cosplay and latex" --tags bdsm lingerie [--fake] [--no-choice]
python main.py eval [--mode navigate|flat|separate] [--no-choice] [--fake] [--output out/report.json]
```

`tag` prints the ranked tags and the navigation trace. Without
`OPENROUTER_API_KEY`, both commands exit with a clear message unless
`--fake` is given.

## Testing

`unittest`, offline, using FakeJudge and a recorded-response stub for
JevJudge:

- request building: shared `state` lists all candidates; one `noul`
  question per candidate; choice criteria keys are the shortlist; request
  splitting at `max_per_request`;
- response parsing, including error bodies and retries;
- navigation on small synthetic trees: pruned branches are never judged,
  uneven depth works, keep rule and floor behave, choice reorders only the
  top `m`;
- taxonomy adapter: 9 top categories, 50 leaves, descriptions skip excluded
  keywords;
- cache: identical request → no second call;
- output rows pass the existing evaluator.

## Left to the builder

Module layout and names; exact prompt wording for the `noul`/`choice`
instructions and criteria (keep the paper's style: "Is the information
needed … in Candidate *n*?"); retry/backoff numbers; report formatting;
whether the fake judge is word overlap or something else deterministic.

## Calls made by Claude (not decided by the user)

- Folder placement next to `adult_ontology_v1`; Python + stdlib + unittest.
- Dropping `adult_content` so the first decision is over 9 categories.
- Declared concepts rank ahead of inferred ones and are not sent to Jev as
  part of the query.
- Using example keywords inside node descriptions.
- The default thresholds and budgets in the table above.
- Choice step reorders only tags that already passed `tag_threshold`, and
  is skipped when fewer than two tags pass.
- Gitignored response cache; MiniLM compared from its committed predictions
  rather than rerun.
- Only level-2 groups (never level-1 categories) may be returned as general
  tags; unjudged children count as "did not pass"; flat/separate ablations
  score the same returnable units as navigation, so they compare fairly.

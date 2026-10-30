# Auto-built keyword tree for the Jev DecisionRAG prototype — design

Date: 2026-10-29
Builds on: `docs/superpowers/specs/2026-10-29-jev-decisionrag-design.md` and the
merged prototype in `AI-Agent-KnowledgeBase/ontology-use-case/Jev-RAG-DecisionModel/`.
Paper: arXiv:2610.35868, Section 3.4 and Appendices A.4 and B.5 (Natural Questions).

## Intent

The prototype so far uses the paper's **first case**: a corpus that already has
a structure. The tree is the user's hand-built taxonomy, which runs 9 categories → groups → 50 concepts.
The user wants to add the paper's **second case**, for corpora with no
structure. The tree is built offline by **k-means over e5 embeddings**, and
an **LLM writes a one-sentence description of each cluster**. The tree is
rebuilt whenever the data changes. This mirrors what the paper did for Natural
Questions (e5-base-v2 on title + first sentence, k-means into 1,000 groups,
Qwen3-8B descriptions).

The user's words: implement the method on *their own* keywords and taxonomy,
not a duplication of the paper. The starting point is `keywords.csv`, taken
as if no taxonomy existed.

The goal is to see how a machine-built tree performs on the same rooms,
next to the hand-built tree, flat scoring, and the existing baselines.

## Goals

1. An **offline tree builder** that reads `adult_ontology_v1/keywords.csv`.
   It embeds each keyword with **e5-base-v2**, clusters the keywords with
   **two-level k-means**, has an **LLM describe every cluster**, and writes
   a **tree file**.
2. A **rebuild command**, `python main.py build-tree`. It is deterministic
   for unchanged input except for the LLM wording, which the tree file pins
   (see Reproducibility).
3. A **keyword-tree adapter**. The engine navigates top clusters →
   sub-clusters → **keywords** for a room, and the **output is keywords**.
4. **Comparison results** (user decision). After a live run, `RESULTS.md`
   gains a section that compares the two workflows side by side on the same
   28 rooms:
   - hand-built taxonomy tree: navigate and flat
   - auto keyword tree: navigate and keyword-flat
   - keyword baseline and MiniLM baseline

   For each, it reports micro-F1, hierarchical micro-F1, precision, recall,
   negative FP rate, Hit@1, latency, cost and requests, plus per-slice notes
   and the unscorable-hit count. The first run's section stays as it is.
5. **Purely additive** (user decision). The new workflow sits beside the
   existing one and replaces nothing. The taxonomy tree, its runs, its
   defaults, `tag`'s default `--tree taxonomy` and every existing command
   and output keep working exactly as now.
6. **Scoring by concepts only** (user decision). Each selected keyword maps
   to its `concept_id` in `keywords.csv`. That concept list is scored on the
   28 test rooms with the existing evaluator, so the results sit beside the
   hand-built tree, flat, keyword and MiniLM runs.

## Non-goals

- No change to the engine (`tree.py`, `judge.py`, `navigate.py`) or to the
  Jev client, beyond what the adapter needs.
- No use of `taxonomy_v1.json` or `concept_id` anywhere in tree building.
  Concepts are used only to score.
- No online or incremental tree updates. Rebuild means rerunning the
  command.
- No tuning of cluster counts or thresholds on the 28 test rooms.
- No change to `adult_ontology_v1/` (it stays read-only).

## Anti-goals

- The taxonomy leaking into the auto tree, for example by embedding concept
  labels or describing clusters with concept names. That would defeat the
  point.
- Heavy ML dependencies (`torch`, `sentence-transformers`, `scikit-learn`,
  the OpenAI client) becoming required for navigation, eval, the CLI, or the
  test suite. They are needed **only** by `build-tree`, imported lazily.
- Tests that load e5, call OpenAI, or need any API key.
- Reporting numbers tuned on the test rooms as if held out.

## Constraints and existing resources

- e5-base-v2 is already in the local Hugging Face cache
  (`models--intfloat--e5-base-v2`). `sentence-transformers`, `torch`,
  `scikit-learn` and `numpy` are installed.
- The LLM for descriptions is **OpenAI** (the Claude-made call the user accepted with
  "lgtm"), using `OPENAI_API_KEY` from the repo-root `.env`. That file is
  already loaded by `main.py`. Call the Chat Completions API over stdlib
  `urllib`, like `jev.py`. A small, current chat model, pinned by exact id
  and recorded in the tree file. The builder picks the id from the account's
  `/v1/models`.
- Jev cannot write descriptions. It only returns decisions.
- `keywords.csv` has 422 rows. Their breakdown:
  - 314 `model_reviewed` with a concept
  - 25 `seed_review_required` with a concept
  - 76 `policy_review_required` with **no concept** (slurs and other policy terms)
  - 7 `excluded` with no concept
  - No keyword appears twice.

## How it works

### Input (Claude's call)

All rows except `review_status = excluded`, which leaves **415 keywords**.
The 76 policy-review keywords stay in. They belong to the corpus, and a room
can genuinely match one. They have no concept, so they cannot score (see
Scoring).

### Embedding

- Model: `intfloat/e5-base-v2`, loaded from the local cache with the
  `passage: ` prefix.
- Normalized embeddings (L2).
- Text embedded: **the keyword string only**. A keyword has no "first
  sentence" the way a page does. Adding its concept label would leak the
  taxonomy.

### Clustering (Claude's call on sizes)

- Two-level k-means, `random_state=0`, `n_init=10`:
  - **Level 1:** 20 clusters over all 415 keywords, about 21 each.
  - **Level 2:** each level-1 cluster with **more than 8 keywords** is split
    into `ceil(size / 5)` sub-clusters, about 5 keywords each. Smaller
    clusters stay one level, so their keywords hang directly under the
    level-1 node. This uneven depth is already supported by the engine.
- One keyword appears in exactly one leaf position.

### Descriptions

- One OpenAI call per cluster and sub-cluster, roughly 20 + 85 calls.
- The prompt is adapted from the paper's B.5 wording: "Write one sentence (at
  most 30 words) describing what the items in this group are about. Name the
  main kinds of subjects it covers, including the less typical ones, so a
  reader can tell whether a given item belongs to this group."
- The input is the cluster's keywords, most central first (closest to the
  centroid), at most 60.
- `temperature=0`.
- If the call errors or the model refuses (some groups contain slurs or
  explicit terms), the description falls back to
  `"Terms such as: kw1, kw2, kw3, …"` (first 8). The node is flagged
  `description_source: "fallback"`, so a rebuild never fails on one cluster.

### Tree file (committed)

`Jev-RAG-DecisionModel/trees/keyword_tree.json` holds:

- **Metadata:**
  - `keywords.csv` SHA-256
  - embedding model id
  - k-means settings and seed
  - LLM model id and prompt
  - build date
  - counts
- **The tree:** nodes with `id`, `description`, `children`, and for keyword
  leaves the `keyword` and `concept_id` (or empty).

Navigation and eval read only this file, never the ML stack.

### Reproducibility

- Embeddings and k-means are deterministic for unchanged input.
- LLM wording can drift between rebuilds. That is why the file is committed,
  so the eval is reproducible from it.
- The eval prints a warning when `keywords.csv`'s current hash differs from
  the one recorded in the tree file ("tree is stale; run build-tree").

### Navigation and output

- `keyword_corpus.py` (adapter) loads the tree file into `tree.Node`s.
  - Cluster nodes: description = the LLM sentence.
  - Keyword leaves: description = the keyword.
  - Only keyword leaves are returnable.
- It uses the same `NavigationConfig` defaults as the taxonomy run, fixed
  before any run:
  - keep a branch at P(yes) ≥ 0.3, at most 4 per level
  - tag at P(yes) ≥ 0.5
  - choice over the top 20
  - binary question wording as in the taxonomy adapter
- The first stage judges all 20 level-1 clusters in one shared context.
- **Output:** ranked keywords with P(yes) and P(mul). `tag` with
  `--tree keywords` prints them, and each keyword shows its concept in
  brackets for readability.
- Self-declared concepts pass straight through, as in the taxonomy adapter.

### Scoring

- Map each ranked keyword to its `concept_id`. Drop keywords with no
  concept. Deduplicate concepts, keeping the first, so the concept order
  follows the best keyword. Self-declared concepts are prepended.
- The resulting concept list is the prediction row for the existing
  evaluator.
- Also record, per room, the raw keyword list and the number of selected
  keywords **with no concept** ("unscorable hits"). Report the total in the
  report, so policy-term hits are visible rather than silently dropped.
- No level normalization (Claude's call). Keyword concepts sit at levels
  1, 2 and 3, while the hand-built tree returns levels 2 and 3. So a
  keyword can map to a parent or child of the gold concept. The existing
  evaluator already reports **hierarchical F1**, which expands
  predictions and gold to their ancestors, next to exact-concept F1. The
  report table gains a **hierarchical micro-F1** column for every run, so
  level mismatches show up there rather than being hidden by a
  normalization step.
- Unscorable hits are reported, and F1 is left unadjusted (Claude's call).
  Each run shows the total number of selected keywords with no concept,
  and how many rooms had at least one.
- New eval runs, added to `--mode all`:
  - `decisionrag-keyword-tree-navigate`
  - `decisionrag-keyword-flat`: all 415 keywords judged with no tree. This
    is a control: is the auto tree helping or hurting compared with no
    tree?
    - It uses **requests of 50** (Claude's call), unlike the taxonomy flat
      run. One context of 415 candidates would be about 90 KB, far over
      Jev's ~32 KB fast-mode limit.
    - Pools split into requests of at least 50 lose nothing for screening
      (paper Section 6.2).
    - So the control is "no tree, same judge" in practice, not "one shared
      context".
- Cost: navigate needs about 4 requests per room and flat 9 per room. That
  is about 365 requests for both runs, a few tenths of a cent. The Jev cache
  is keyed by request body, so a rebuilt tree with new descriptions misses
  the cache and is billed once. This is expected.
- The existing runs stay unchanged.

### CLI

- `python main.py build-tree [--clusters 20] [--leaf-size 5] [--out trees/keyword_tree.json]`
  - Requires `OPENAI_API_KEY`.
  - Prints cluster sizes, the number of fallback descriptions, and the
    OpenAI token usage.
  - The OpenAI model is a **hardcoded default constant** (Claude's call). The
    builder picks one small, current chat model once, from the account's
    `/v1/models`, while implementing. `--llm-model` overrides it. The id
    actually used is recorded in the tree file.
  - `build-tree --fake` uses a deterministic fake embedder and a fake
    describer, and writes to a scratch path unless `--out` is given. It is
    a smoke test that needs no key and no ML libraries.
- `python main.py tag ... --tree taxonomy|keywords` (default `taxonomy`).
- `python main.py eval --mode all` includes the two new runs when the tree
  file exists.
  - If the tree file is missing, they are skipped with a note.
  - A stale tree (the `keywords.csv` hash changed) still runs, with a
    warning.
  - `tag --tree keywords` with no tree file exits with a message pointing to
    `build-tree`.
- Key checks are per command (Claude's call). `main()` currently checks for a
  Jev key before dispatching any command. That check moves so that:
  - `build-tree` needs only `OPENAI_API_KEY`
  - `tag`, `demo` and `eval` need a Jev key
  - `--fake` needs neither

## Testing

- Offline and hermetic, as now.
- The builder takes an **injectable embedder and describer**. Tests use a
  fake embedder (deterministic vectors) and a fake describer, so e5 and
  OpenAI are never loaded.
- Tests cover:
  - tree shape and the leaf-size rule
  - excluded rows dropped
  - no concept text in any embedded string or prompt
  - fallback description on describer error
  - tree file round-trip
  - stale-hash warning
  - keyword → concept mapping, deduplication and the unscorable count
  - eval rows accepted by the existing evaluator
- No test imports `torch`, `sentence_transformers`, `sklearn`, or calls
  OpenAI.

## Left to the builder

- Module names beyond `keyword_corpus.py`.
- Exact OpenAI model id (small, current, pinned).
- How the centroid ordering is computed.
- Report formatting of the new runs.
- Retry details for OpenAI calls (reuse `jev.py`'s pattern).

## Calls made by Claude (not decided by the user)

- OpenAI for descriptions, because the key is already in `.env`.
- Keep the 76 policy-review keywords in the tree. Drop the 7 excluded.
- Embed the keyword string only (no concept label).
- Two levels: 20 top clusters, then splits of about 5 keywords for clusters
  over 8.
- Fallback description on refusal or error, instead of failing the build.
- Commit the tree file. Warn on a stale `keywords.csv`.
- Add the `keyword-flat` control run, with 50 candidates per request
  because of Jev's context limit.
- Same navigation defaults as the taxonomy run, with no tuning.
- No level normalization. Add a hierarchical-F1 column for all runs.
- Report unscorable hits, and leave F1 unadjusted.
- Hardcoded, overridable OpenAI model id. `build-tree --fake` for smoke
  runs.
- Key checks per command. A missing tree file skips the new eval runs, and
  a stale one warns.

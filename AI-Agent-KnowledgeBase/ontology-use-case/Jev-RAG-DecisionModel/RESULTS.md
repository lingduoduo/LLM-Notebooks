# First live results (2026-10-29)

Run: `python3 main.py eval --mode all` against TypeSafe's direct API, model
`jev-latest` (resolved to **jev-1.14.0**). 28 test cases from
`adult_ontology_v1/evaluation/semantic_cases.json`, including 10 negatives.
Every threshold and budget kept the value fixed before the run (branches
kept at P(yes) ≥ 0.3, at most 4 per level; tag at P(yes) ≥ 0.5; choice over
the top 20). **Nothing was tuned on these cases.**

| run | micro-F1 | precision | recall | macro-F1 | exact match | negative FP rate | Hit@1 | MRR | latency (s/room) | est. cost (¢) | requests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DecisionRAG, navigate | 0.452 | 0.33 | 0.73 | 0.333 | 0.321 | 0.40 | 0.722 | 0.750 | 0.50 | 0.30 | 102 |
| DecisionRAG, navigate, no choice | 0.452 | 0.33 | 0.73 | 0.333 | 0.321 | 0.40 | 0.778 | 0.778 | (cached) | 0.26 | 84 |
| DecisionRAG, flat (one context) | **0.571** | 0.41 | **0.92** | 0.426 | 0.286 | 0.90 | 0.889 | 0.917 | 0.23 | 0.58 | 44 |
| DecisionRAG, separate | 0.240 | 0.14 | 0.92 | 0.158 | 0.179 | 0.50 | 0.944 | 0.944 | 1.18 | 3.36 | 2065 |
| Keyword baseline | **0.679** | **0.67** | 0.69 | 0.460 | 0.536 | 0.80 | – | – | – | 0 | 0 |
| MiniLM baseline | 0.383 | 0.43 | 0.35 | 0.275 | 0.357 | **0.30** | 0.778 | 0.815 | – | 0 | 0 |

Estimated cost is input tokens × $0.042 per 1M. TypeSafe reported no cost.
The whole run made 2,204 uncached requests, about 4.3¢ in total. The
no-choice run reused the navigate run's cached binary requests, so its
latency is not meaningful.

## What it shows

- **Recall is high and precision is low.** Jev finds most gold concepts
  (navigate 0.73, flat 0.92 recall, against 0.69 for keywords and 0.35 for
  MiniLM), but it also tags closely related concepts. For example, "Private
  show tonight." also gets `interactive_show` and `paid_session`, and
  "BDSM themed room." gets all of `bdsm`'s children rather than `bdsm`
  itself. The fixed tag threshold of 0.5 is too low for how confident Jev
  is: related concepts score 0.6–0.9.
- **Meaning beats matching where keywords fail.** On the paraphrase slice,
  navigate scores 0.60 and flat 0.70, against 0.00 for keywords and 0.40 for
  MiniLM. On generic language, navigate scores 0.67 and both baselines 0.
- **Flat beats navigation here, unlike the paper.** With only 73 returnable
  concepts, everything fits in one shared context (16.6 KB). Pruning at the
  category level then only costs recall (0.73 against 0.92). Navigation pays
  off when the corpus is too big for one context, as in the paper's 2.68M
  passages. This taxonomy is not.
- **A shared context matters, as the paper found.** Separate per-candidate
  requests have the same recall as flat but much worse precision (0.14 against
  0.41), at 47× the requests and 6× the cost. Judging candidates side by side
  makes the probabilities comparable.
- **The choice step did not help ordering here.** Hit@1 was 0.722 with it and
  0.778 without. The tag sets are small (a handful of tags), so there is
  little head to sharpen.
- **Negation, word boundary, educational and benign-context cases fail for
  every system** (micro-F1 0). Jev still tags concepts that are only
  mentioned, despite the "negated or unrelated counts as no" rule in the
  shared context.
- **Identity inference:** as anticipated in the design, Jev inferred
  `gender_identity` on the `neutral-identity` negative case.

## Next steps worth trying

1. Raise the tag threshold, or keep only the choice step's top picks. This
   must be done on held-out cases, or clearly labelled "tuned on test".
2. Make the stop-at-general-node rule stricter, so that `bdsm` is returned
   when its children are only partly supported.
3. Put the negation rule into each question's `false` criterion again (it
   was moved into the shared state to fit the flat request) and measure the
   effect.

# Comparison: hand-built taxonomy tree vs auto keyword tree (2026-10-30)

Here two workflows run side by side on the same 28 test rooms. The judge is
Jev (`jev-latest`, resolved **jev-1.14.0**) through TypeSafe's API, with the
same fixed navigation settings and nothing tuned on these cases.

- **Hand-built taxonomy tree:** your `taxonomy_v1.json`, 9 categories → about
  23 groups → 50 concepts. Its numbers are the first run's above; the
  requests came from the cache.
- **Auto keyword tree:** built by `python main.py build-tree` from
  `keywords.csv` alone, without the taxonomy. It holds 415 keywords, e5-base-v2
  embeddings, and k-means into 20 clusters, which are split into 89
  sub-clusters (level-1 sizes run from 7 to 35). The keywords that Jev selects
  are mapped to their concepts only for scoring. The tree is committed at
  `trees/keyword_tree.json`.

> **Caveat: there are no LLM descriptions.** The OpenAI key in `.env` was
> rejected (401), so all 109 cluster descriptions are keyword-list fallbacks
> ("Terms such as: …"). The tree file records this
> (`describer_status: rejected`, `llm_model: none`). These numbers are a
> floor for the auto tree, and real one-sentence descriptions could change
> them. With 28 rooms, a difference of a few points is noise.

| run | micro-F1 | hier. micro-F1 | precision | recall | negative FP rate | Hit@1 | latency (s/room) | est. cost (¢) | requests |
|---|---|---|---|---|---|---|---|---|---|
| Taxonomy tree, navigate | 0.452 | **0.602** | 0.33 | 0.73 | 0.40 | 0.722 | 0.50 | 0.30 | 102 |
| Taxonomy, flat (one context) | **0.571** | **0.673** | 0.41 | **0.92** | 0.90 | 0.889 | 0.23 | 0.58 | 44 |
| **Auto keyword tree, navigate** | 0.481 | 0.547 | **0.46** | 0.50 | **0.20** | 0.867 | 0.58 | 0.34 | 75 |
| Keywords, flat (50 per request) | 0.407 | 0.520 | 0.26 | 0.88 | 0.50 | 0.944 | 0.52 | 2.48 | 269 |
| Keyword baseline | **0.679** | **0.671** | **0.67** | 0.69 | 0.80 | – | – | 0 | 0 |
| MiniLM baseline | 0.383 | 0.480 | 0.43 | 0.35 | 0.30 | 0.778 | – | 0 | 0 |

The taxonomy rows' latency comes from the first live run, because this run
served them from the cache. The keyword runs were new: 344 uncached requests,
about 2.8¢.

**Concept micro-F1 by slice**, for the slices where the workflows differ:

| slice | taxonomy navigate | taxonomy flat | auto tree navigate | keyword flat | keyword baseline | MiniLM |
|---|---|---|---|---|---|---|
| exact | 0.333 | 0.615 | 0.500 | 0.417 | 1.000 | 0.462 |
| paraphrase | 0.600 | **0.700** | 0.429 | 0.381 | 0.000 | 0.400 |
| generic language | 0.667 | 0.500 | **1.000** | 0.667 | 0.000 | 0.000 |
| multilabel | 0.375 | 0.667 | 0.250 | 0.444 | **1.000** | 0.222 |
| unicode | 0.000 | 0.667 | **1.000** | 0.667 | 1.000 | 0.000 |
| tags | **1.000** | **1.000** | 0.667 | 0.500 | 1.000 | 0.800 |

Negation, word-boundary, educational, benign-context and neutral-identity
cases still score 0 for every system.

## What the comparison shows

- **The machine-built tree holds its own, even without LLM descriptions.** On
  exact concepts it slightly beats the hand-built tree (0.481 vs 0.452
  micro-F1). It is more precise (0.46 vs 0.33) and has the lowest negative
  false-positive rate of any system (0.20). Its recall is lower (0.50 vs
  0.73). Fallback descriptions name concrete keywords, so Jev prunes harder
  and tags less.
- **Structure still matters for meaning.** The hand-built tree is ahead on
  hierarchical F1 (0.602 vs 0.547) and on paraphrase (0.600 vs 0.429). Human
  labels like "Performance Format > Broadcast Mode" help with rooms that
  describe something in other words. A list of example keywords helps less.
  This is the gap that real LLM cluster descriptions should narrow.
- **On keywords, the tree helps, unlike on the taxonomy.** Navigating the
  auto tree beats judging all 415 keywords flat (0.481 vs 0.407 micro-F1).
  It uses 75 requests against 269, at about 1/7 of the cost. On the 73-concept
  taxonomy, flat won because everything fit in one context. With 415 leaves,
  flat has to split into 9 requests and loses its shared context, so the
  tree's pruning pays off. This matches the paper's argument that navigation
  earns its keep once the corpus no longer fits in one request.
- **Recall is spent in different places.** Keyword flat has high recall and
  low precision (0.88 / 0.26), which is the same over-tagging pattern as
  taxonomy separate mode. Without one shared context, Jev's P(yes) values are
  not comparable across requests.
- **Unscorable hits are rare.** Only one selected keyword across all runs had
  no concept (keyword flat, 1 room). Policy-review terms seldom win.

## Next steps worth trying

1. **Add a working OpenAI key and rerun** `python main.py build-tree` followed
   by `python main.py eval --mode all`. The taxonomy runs come from the cache,
   so only the keyword runs cost money. This is the comparison the design was
   meant to measure: real LLM descriptions against the hand-built labels.
2. **Try a coarser split** (`--clusters 10`) or **keep more branches per
   level**. The auto tree's recall loss comes from pruning, as in the paper's
   Figure 7. Report this as tuned on test.

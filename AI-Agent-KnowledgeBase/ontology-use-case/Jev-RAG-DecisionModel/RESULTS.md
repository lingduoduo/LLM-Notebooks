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

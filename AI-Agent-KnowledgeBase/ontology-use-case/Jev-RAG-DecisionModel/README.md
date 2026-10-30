# DecisionRAG prototype (Jev)

A small, stdlib-only Python prototype of **DecisionRAG** ("Retriever-Free Retrieval: Navigating Documents with Decision Models", [arXiv:2610.35868](https://arxiv.org/abs/2610.35868)).

## Core idea

- No whole-corpus similarity search: walk the corpus tree level by level.
- At each level one decision-model (Jev) pass judges all candidates together in a shared context and keeps the promising branches; pruned branches are never expanded.
- An optional multiple-choice pass reorders the head of the final shortlist.

## Setup

Python 3.10+, no dependencies.

Put a key in the repo-root `.env` (gitignored) or export it:

```
TYPESAFE_API_KEY=...      # preferred: calls TypeSafe directly (https://api.typesafe.ai/v1/systemone)
OPENROUTER_API_KEY=...    # fallback: OpenRouter, model typesafe/jev-1.13
```

`TYPESAFE_API_KEY` wins if both are set; variables already in the environment are not overridden by `.env`. The TypeSafe provider uses the unpinned alias `jev-latest`; each response reports the resolved version (e.g. `jev-1.13.0`), which is printed after `tag`/`demo` runs and recorded in `out/report.json` and `out/report.md`. TypeSafe returns no cost, so cost is an **estimated cost** (input tokens x $0.042 per 1M).

Add `--fake` to any command to use an offline lexical stand-in judge (no key, no network; useful for smoke tests, not for quality).

## Commands

```
python3 main.py tag "bdsm dungeon" --tags bdsm --declared orientation --mode navigate [--no-choice]
python3 main.py eval --mode all [--output out/report.json]
python3 main.py demo
```

- `tag SUBJECT`: tags one room against the adult-content taxonomy; prints `id p_bin p_mul source`, then each stage's trace. `--mode` is `navigate`, `flat` (every unit judged in one shared-context request) or `separate` (one request per unit). With the real judge it also prints the provider, resolved model version(s), the nominal (or estimated) cost, requests, and cache hits/misses.
- `eval`: runs the labeled test rooms. `--mode all` runs navigate, navigate without choice, flat, separate, plus the keyword and MiniLM baselines, and writes `out/report.json` and `out/report.md` (per-run micro/macro-F1, exact match, negative FP rate, Hit@1, MRR, mean latency per room, nominal cost, requests). If a Jev call fails mid-eval, the CLI prints `error: ...`, writes the report for the runs already finished, and exits 1.
- `demo`: the paper's running example ("Who is the main character in Christopher Nolan's film from the year 2000?") on a document -> passage tree in `examples/nolan_corpus.json`. Prints each stage's P(yes), kept and pruned branches, and the final passage ranking.

## Tests

```
python3 -m unittest discover -v
```

Tests are offline and need no key.

## Cache

Every Jev response is cached under `.cache/jev/<provider>/` (keyed by the request), so rerunning an identical eval costs nothing; responses that fail to parse are never cached. After a real run the CLI prints cache hits and misses. Reported cost is a **nominal cost (cached responses included; real spend = cache misses)**. `.cache/` and `out/` are gitignored.

## Plugging in another corpus

Build a tree of `tree.Node(id, description, children)` where each description is what the judge sees (a title for a document, the text for a passage), then pick a `NavigationConfig` (`k`, thresholds, `binary_question` containing `{n}`, and `returnable` to say which nodes can be answers) and call `navigate(root, query, judge, config)`. `main.py`'s `load_demo_tree` is a ten-line example.

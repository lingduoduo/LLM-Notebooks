# Jev DecisionRAG Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A corpus-agnostic DecisionRAG engine (paper arXiv:2610.35868, Algorithm 1) that navigates a tree with Jev decisions, applied first to tagging rooms with `adult_ontology_v1` taxonomy concepts and evaluated against that project's baselines.

**Architecture:** A generic engine (`tree.py`, `judge.py`, `navigate.py`) knows only nodes with descriptions and a judge that returns P(yes) per candidate and choice probabilities. `jev.py` implements the judge over OpenRouter's Decisions API, with a disk cache at the transport layer. `taxonomy_corpus.py` adapts the taxonomy into a tree and adds self-declared pass-through; `evaluate.py` scores runs with the ontology project's own evaluator; `main.py` is the CLI.

**Tech Stack:** Python 3.12, stdlib only (`urllib.request`, `concurrent.futures`, `dataclasses`, `hashlib`, `json`, `argparse`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-29-jev-decisionrag-design.md`

All paths below are relative to `AI-Agent-KnowledgeBase/ontology-use-case/Jev-RAG-DecisionModel/` unless they start with `adult_ontology_v1/` (sibling directory) or `docs/`. Run tests from the prototype directory: `python3 -m unittest discover -v`.

## Global Constraints

- Stdlib only; HTTP via `urllib.request`; tests are `unittest`; no test may make a network call or need `OPENROUTER_API_KEY`.
- `adult_ontology_v1/` is read-only. Import from it by inserting its directory (`Path(__file__).resolve().parent.parent / "adult_ontology_v1"`) at the front of `sys.path`; use `utils.load_taxonomy`, `utils.load_keyword_rows`, `utils.load`, `utils.RoomClassifier`, `evaluation.semantic_evaluation.evaluate`, `load_benchmark`.
- The engine (`tree.py`, `judge.py`, `navigate.py`, `jev.py`) contains no taxonomy, room, or concept-ID knowledge and no embedding/similarity code.
- Endpoint `https://openrouter.ai/api/alpha/decisions`; model pinned to `"typesafe/jev-1.13"`; key from `OPENROUTER_API_KEY`.
- Defaults: `keep_threshold=0.3`, `k=4`, `floor=0.1`, `tag_threshold=0.5`, `m=20`, `max_per_request=50`, at most 8 requests in flight.
- `out/` and `.cache/` are gitignored.
- Headline numbers use the fixed defaults; any threshold sweep is labeled "tuned on test, not held out".

## Review Focus

1. Jev returns an error body, a non-200 status, or an `answers` map missing a question key → a `JevError` naming the status/missing key, never a bare `KeyError` (test in Task 3).
2. A room with an empty subject and no tags → the query still builds (`Tags: (none)`), navigation runs, and the result may be empty without crashing (test in Task 4).
3. Non-ASCII room text (the `unicode` slice) → the request JSON keeps the characters and the cache key is the same for the same request regardless of dict key order (test in Task 3).
4. Every top-level P(yes) below `floor` → nothing is expanded, no choice call is made, and the result is empty (test in Task 2).
5. A choice answer that omits a shortlisted option → that option gets probability 0 and keeps its binary order among ties (test in Task 2).

---

### Task 1: Tree, judge interface, and fake judge

**Files:**
- Create: `.gitignore` (`out/`, `.cache/`)
- Create: `tree.py`, `judge.py`
- Test: `test_judge.py`

**Interfaces:**
- Produces:
  - `tree.Node` — frozen dataclass: `id: str`, `description: str`, `children: tuple[Node, ...] = ()`, `payload: Mapping = field(default_factory=dict)`; property `is_leaf -> bool`.
  - `tree.walk(root: Node) -> Iterator[Node]` — pre-order, root excluded.
  - `judge.Judge` — `typing.Protocol` with `binary(query: str, candidates: Sequence[Node], question: str) -> dict[str, float]` and `choice(query: str, candidates: Sequence[Node], question: str) -> dict[str, float]`.
  - `judge.FakeJudge(scores: Mapping[str, float] | None = None)` — if `scores` is given, `binary` returns `scores.get(id, 0.0)`; otherwise word overlap: fraction of the description's lowercase word set that appears in the query's word set. `choice` returns the binary scores normalized to sum 1 (uniform if all zero). Records every call in `self.calls: list[tuple[str, tuple[str, ...]]]` as `("binary"|"choice", candidate_ids)`.

- [ ] **Step 0: Create the branch.** From the current `refactor/utils-and-main` HEAD (it carries the `utils.py` the adapter imports): `git switch -c feat/jev-decisionrag`.
- [ ] **Step 1: Write failing tests** in `test_judge.py`:
  - `test_walk_is_preorder`: tree `r(a(a1,a2),b)` → `[n.id for n in walk(r)] == ["a","a1","a2","b"]`.
  - `test_fake_scores_lookup`: `FakeJudge({"a": .9}).binary("q", [a, b], "?") == {"a": .9, "b": 0.0}`.
  - `test_fake_overlap`: query `"latex outfit"`, description `"Latex"` → 1.0; description `"Gaming stream"` → 0.0.
  - `test_fake_choice_sums_to_one`: `assertAlmostEqual(sum(FakeJudge({"a": .6, "b": .2}).choice("q", [a, b], "?").values()), 1.0)`, and `"a"` > `"b"`.
  - `test_fake_records_calls`: after one binary call on `[a, b]`, `calls == [("binary", ("a", "b"))]`.
- [ ] **Step 2: Run** `python3 -m unittest test_judge -v` → FAIL (`ModuleNotFoundError: tree`).
- [ ] **Step 3: Implement** `tree.py` and `judge.py` per the Interfaces block.
- [ ] **Step 4: Run** `python3 -m unittest test_judge -v` → PASS.
- [ ] **Step 5: Commit** `git add .gitignore tree.py judge.py test_judge.py && git commit -m "feat(decisionrag): add tree, judge protocol, and fake judge"`

### Task 2: Navigation engine (Algorithm 1) and flat/separate modes

**Files:**
- Create: `navigate.py`
- Test: `test_navigate.py`

**Interfaces:**
- Consumes: `Node`, `walk`, `Judge`, `FakeJudge` (Task 1).
- Produces:
  - `navigate.NavigationConfig` — frozen dataclass: `keep_threshold=0.3`, `k=4`, `floor=0.1`, `tag_threshold=0.5`, `r: int | None = None` (paper Step 2: after the threshold, keep only the top `r` units by P(yes); `None` = no cap. A document corpus uses the paper's pure top-r with `tag_threshold=0.0, r=…`; the taxonomy uses the threshold and no cap), `m=20`, `use_choice=True`, `binary_question: str`, `choice_question: str`, `returnable: Callable[[Node], bool] = lambda n: n.is_leaf`. Question defaults are generic wording ("Is Candidate {n} relevant to the query?" / "Which candidate is most relevant to the query?"); `{n}` is filled by the judge.
  - `navigate.Ranked` — dataclass: `id: str`, `p_bin: float | None`, `p_mul: float | None = None`, `source: str = "inferred"`.
  - `navigate.Result` — dataclass: `ranked: list[Ranked]`, `trace: list[dict]`, `latency_s: float`.
  - `navigate.navigate(root: Node, query: str, judge: Judge, config: NavigationConfig) -> Result`.
  - `navigate.score_units(units: Sequence[Node], query: str, judge: Judge, config: NavigationConfig, *, separate: bool = False) -> Result` — flat (one `binary` call over all units) or separate (one call per unit), then the same tagging and choice step.

Algorithm (the parts the signature doesn't fix):

```
C = root.children; scores = {}
while C:
    p = judge.binary(query, C, config.binary_question); scores.update(p)
    internal = [c for c in C if not c.is_leaf], sorted by p desc (stable)
    keep = [c for c in internal if p[c] >= keep_threshold][:k]
    if not keep and internal and p[internal[0]] >= floor: keep = [internal[0]]
    trace.append({"judged": [ids], "p": p, "kept": [ids], "pruned": [ids]})
    C = [child for c in keep for child in c.children]
units = every node judged with config.returnable(node) true
tag internal unit u iff scores[u] >= tag_threshold and no child c has scores.get(c, 0) >= tag_threshold
tag leaf u iff scores[u] >= tag_threshold
tags sorted by p_bin desc (stable, tree order on ties); if r is not None: tags = tags[:r]
if use_choice and len(tags) >= 2: q = judge.choice(query, tags[:m] nodes, choice_question);
    reorder tags[:m] by q.get(id, 0) desc (stable); set p_mul
```

- [ ] **Step 1: Write failing tests** in `test_navigate.py`, on synthetic tree `root(A(a1,a2), B(b1(b1x,b1y)), c)` (uneven depth; `c` a top-level leaf), using `FakeJudge(scores)`:
  - `test_pruned_branch_never_judged`: scores `{A:.9,a1:.8,B:.05}` → no call's ids contain `b1`, `b1x`, `b1y`.
  - `test_uneven_depth_returns_leaves_at_any_level`: scores `{A:.9,a1:.8,c:.7,B:.9,b1:.9,b1y:.6}` → tag ids == `{"a1","c","b1y"}`.
  - `test_keep_cap_k`: five internal top-level children all at .9, `k=4` → trace[0]["kept"] has 4 ids.
  - `test_floor_keeps_top_one`: `{A:.2,B:.15}` → kept `["A"]`; `test_below_floor_stops` (Review Focus 4): `{A:.05,B:.05,c:.05}` → `ranked == []`, no choice call in `calls`.
  - `test_internal_node_returnable_when_no_child_passes`: `returnable=lambda n: n.is_leaf or n.id=="A"`, scores `{A:.9,a1:.2,a2:.1}` → tags `["A"]`; with `a1:.8` → tags `["a1"]`.
  - `test_choice_reorders_only_top_m`: `m=2`, tags with p_bin a1 .9, a2 .8, c .7, choice `{a2:.7,a1:.3}` (stub judge subclassing FakeJudge overriding `choice`) → order `["a2","a1","c"]`, `c.p_mul is None`.
  - `test_choice_skipped_for_single_tag` and `test_choice_missing_option_is_zero` (Review Focus 5): choice returns `{a2: 1.0}` only for shortlist `[a1,a2,c]` → `a2` first, then `a1`, `c` in binary order, `a1.p_mul == 0.0`.
  - `test_score_units_separate_calls_once_per_unit`: 3 units, `separate=True` → 3 binary calls of length 1.
  - `test_top_r_caps_units`: four leaves passing threshold, `r=2` → exactly the two highest returned.
  - `test_paper_nolan_example` (the paper's Figure 2c/3, document → passage): tree `root(nolan(n1,n2), memento(m1,m2,m3), gladiator(g1,g2))`, config `k=2, tag_threshold=0.0, r=3, m=2`, scores `{nolan:.81, memento:.96, gladiator:.07, m1:.9 (Leonard Shelby passage), n2:.6, m2:.4, …}`, choice stub preferring `m1` → `gladiator`'s passages `g1,g2` never appear in any call (pruned despite keyword overlap), returned ids are 3 passages from `nolan`/`memento`, first is `m1`.
- [ ] **Step 2: Run** `python3 -m unittest test_navigate -v` → FAIL (`ModuleNotFoundError: navigate`).
- [ ] **Step 3: Implement** `navigate.py` per the Interfaces and algorithm; `latency_s` via `time.perf_counter`.
- [ ] **Step 4: Run** `python3 -m unittest test_navigate -v` → PASS.
- [ ] **Step 5: Commit** `git add navigate.py test_navigate.py && git commit -m "feat(decisionrag): add hierarchical navigation and flat/separate scoring"`

### Task 3: Jev judge over OpenRouter, with disk cache

**Files:**
- Create: `jev.py`
- Test: `test_jev.py`

**Interfaces:**
- Consumes: `Node`, `Judge` protocol (Task 1).
- Produces:
  - `jev.ENDPOINT`, `jev.MODEL = "typesafe/jev-1.13"`, `class JevError(RuntimeError)`.
  - `jev.build_binary_request(query: str, candidates: Sequence[Node], question: str) -> dict` — `state = {"query": query, "candidates": "1. <desc>\n2. <desc>…"}`; `questions = {"c1": {"type": "noul", "instructions": question.format(n=1), "criteria": {"true": …, "false": …}}, …}` (criteria wording is the builder's; it must say a mention that is negated or unrelated counts as false).
  - `jev.build_choice_request(query, candidates, question) -> dict` — same `state`; one question `"most_relevant"` of type `choice` with `criteria = {node.id: node.description}`.
  - `jev.parse_binary(response: dict, candidates) -> dict[str, float]`, `jev.parse_choice(response: dict, candidates) -> dict[str, float]` (missing options → 0.0; missing question key → `JevError`).
  - `jev.HttpTransport(api_key: str, timeout: float = 60)` — callable `(body: dict) -> dict`; POST with `Authorization: Bearer`, JSON `ensure_ascii=False` UTF-8; retries 429/5xx/`URLError` up to 3 times with backoff 1 s, 2 s, 4 s; other non-200 → `JevError(f"{status}: {message}")`.
  - `jev.CachedTransport(inner, cache_dir: Path)` — key `sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode())`; stores raw response JSON; `hits: int`, `misses: int`.
  - `jev.JevJudge(transport, model: str = MODEL, max_per_request: int = 50, max_workers: int = 8)` — implements `Judge`; splits candidates into chunks of `max_per_request` sent in parallel via `ThreadPoolExecutor`; adds `"model"` to each body; accumulates `requests: int` and `cost_usd: float` from `usage.cost`.

- [ ] **Step 1: Write failing tests** in `test_jev.py` with a fake transport (a callable recording bodies and returning canned responses shaped like the API reference):
  - `test_binary_request_shares_state`: 3 candidates → one body, `state["candidates"]` contains all three numbered descriptions, `questions` keys `{"c1","c2","c3"}`, each `type == "noul"` with `"true"`/`"false"` criteria.
  - `test_choice_request_keys_are_ids`: criteria keys equal candidate ids.
  - `test_split_at_max_per_request`: 120 candidates, `max_per_request=50` → 3 bodies of sizes 50, 50, 20; merged result has 120 ids.
  - `test_parse_binary` (`{"answers": {"c1": {"type":"noul","noul":.96}}}` → `{"a": .96}`) and `test_parse_missing_key_raises_jev_error` (Review Focus 1).
  - `test_http_non_200_raises_jev_error` and `test_http_retries_429_then_succeeds` — patch `urllib.request.urlopen` and `time.sleep`.
  - `test_cache_hit_skips_inner` and `test_cache_key_ignores_dict_order_and_keeps_unicode` (Review Focus 3): query `"Café — 夜"` → second identical call is a hit; body bytes contain `"Café"`.
  - `test_cost_accumulates`: two responses with `usage.cost` .001 and .002 → `cost_usd ≈ .003`, `requests == 2`.
- [ ] **Step 2: Run** `python3 -m unittest test_jev -v` → FAIL.
- [ ] **Step 3: Implement** `jev.py` per the Interfaces block.
- [ ] **Step 4: Run** `python3 -m unittest test_jev -v` → PASS.
- [ ] **Step 5: Commit** `git add jev.py test_jev.py && git commit -m "feat(decisionrag): add Jev decisions judge with cache and retries"`

### Task 4: Taxonomy adapter

**Files:**
- Create: `taxonomy_corpus.py`
- Test: `test_taxonomy_corpus.py`

**Interfaces:**
- Consumes: `Node` (Task 1); `NavigationConfig`, `Ranked`, `Result`, `navigate`, `score_units` (Task 2); `utils.load_taxonomy`, `utils.load_keyword_rows` from `adult_ontology_v1`.
- Produces:
  - `taxonomy_corpus.ONTOLOGY_DIR: Path` (inserted into `sys.path` at import).
  - `build_tree(taxonomy: dict, keyword_rows: Iterable[dict], max_examples: int = 8) -> Node` — virtual root `"__root__"`; children are the 9 level-1 nodes in file order (level-0 `adult_content` dropped, its children promoted); node `payload = {"level": int}`; description `"<Label> — <Path> (e.g. kw1, kw2, …)"` where Path joins labels from level 1 down with `" > "` and examples are the first `max_examples` keywords with that `concept_id` and `review_status != "excluded"` (no `(e.g. …)` when none).
  - `is_returnable(node: Node) -> bool` — leaf, or `payload["level"] == 2`.
  - `room_query(subject: str, tags: Sequence[str]) -> str` — `"Subject: <subject>\nTags: <first five joined by ', '>"`, `"Tags: (none)"` when empty.
  - `taxonomy_config(**overrides) -> NavigationConfig` — defaults plus `returnable=is_returnable` and room-specific question wording ("Does this room involve Candidate {n}?", "Which candidate is most relevant to this room?").
  - `tag_room(root, judge, config, subject, tags, self_declared=(), mode="navigate") -> Result` — mode `"navigate"` | `"flat"` | `"separate"` (flat/separate pass `[n for n in walk(root) if is_returnable(n)]` to `score_units`); then prepends `Ranked(id, p_bin=None, source="self_declared")` for each declared concept, dropping that id from the inferred list.
  - `load_tree() -> tuple[dict, Node]` — `(taxonomy, build_tree(load_taxonomy(), load_keyword_rows()))`.

- [ ] **Step 1: Write failing tests** in `test_taxonomy_corpus.py` against the real files:
  - `test_top_level_is_nine_categories`: `len(root.children) == 9`, `"adult_content"` not in any node id.
  - `test_fifty_leaves_and_returnable_groups`: 50 leaves; `is_returnable` true for `"bdsm"` and `"roleplay"`, false for any level-1 node.
  - `test_description_has_path_and_examples`: lingerie node description starts with `"Lingerie — Body & Appearance > "` and contains `"e.g."`; no excluded keyword (take them from `keywords.csv` rows with `review_status == "excluded"`) appears in any description.
  - `test_room_query_empty` (Review Focus 2): `room_query("", []) == "Subject: \nTags: (none)"`, and `tag_room(root, FakeJudge(), taxonomy_config(), "", [])` returns a `Result` without raising.
  - `test_room_query_first_five_tags`: seven tags → only the first five appear.
  - `test_self_declared_first_and_deduped`: `FakeJudge({...})` that tags `"orientation"` and `"lingerie"`, `self_declared=["orientation"]` → `ranked[0] == Ranked("orientation", None, None, "self_declared")`, `"orientation"` appears once.
- [ ] **Step 2: Run** `python3 -m unittest test_taxonomy_corpus -v` → FAIL.
- [ ] **Step 3: Implement** `taxonomy_corpus.py` per the Interfaces block.
- [ ] **Step 4: Run** `python3 -m unittest test_taxonomy_corpus -v` → PASS.
- [ ] **Step 5: Commit** `git add taxonomy_corpus.py test_taxonomy_corpus.py && git commit -m "feat(decisionrag): add taxonomy corpus adapter"`

### Task 5: Evaluation and baseline comparison

**Files:**
- Create: `evaluate.py`
- Test: `test_evaluate.py`

**Interfaces:**
- Consumes: `tag_room`, `load_tree`, `taxonomy_config` (Task 4); `semantic_evaluation.evaluate`, `load_benchmark`; `utils.RoomClassifier`, `utils.load`; `adult_ontology_v1/evaluation/local_semantic_predictions.jsonl`.
- Produces:
  - `prediction_row(case: dict, result: Result, taxonomy_version: str) -> dict` — `{"case_id", "taxonomy_version", "concept_ids": [ranked ids], "scores": {id: p_bin for inferred}, "ranked": [asdict(r)…], "trace", "latency_s"}`.
  - `rank_metrics(cases: list[dict], rows: list[dict]) -> dict` — over cases with non-empty gold and non-empty `concept_ids`: `hit_at_1` (first id in gold), `mrr` (1/rank of first gold id, 0 if none), `count`.
  - `run_decisionrag(judge, config, mode="navigate") -> dict` — runs all test cases, returns `{"name", "report": evaluate(...predictions=rows), "rank": rank_metrics, "rows", "cost_usd", "requests"}` (cost/requests from the judge if it has them, else 0).
  - `run_baselines() -> list[dict]` — keyword baseline via `evaluate(classifier=RoomClassifier(taxonomy, load()[1]))` (rank metrics `None`); MiniLM via its committed predictions file (rank metrics from `concept_ids` order).
  - `write_report(runs: list[dict], out_path: Path) -> None` — JSON of all runs at `out_path`, and a Markdown table at `out_path.with_suffix(".md")` with columns: name, concept micro-F1, macro-F1, exact match, negative FP rate, Hit@1, MRR, cost (¢), requests; plus per-slice micro-F1 rows.

- [ ] **Step 1: Write failing tests** in `test_evaluate.py`:
  - `test_prediction_rows_pass_existing_evaluator`: `run_decisionrag(FakeJudge(), taxonomy_config())` returns a report with `count == 28` (the evaluator validates IDs and version itself).
  - `test_rank_metrics`: gold `{"a"}`, predictions `["b","a"]` → `hit_at_1 == 0`, `mrr == 0.5`; empty-gold cases excluded from `count`.
  - `test_baselines_load`: `run_baselines()` names `["keyword", "minilm"]`, each report `count == 28`.
  - `test_write_report_markdown`: writes both files to a temp dir; the `.md` contains a header row with `"Hit@1"`.
- [ ] **Step 2: Run** `python3 -m unittest test_evaluate -v` → FAIL.
- [ ] **Step 3: Implement** `evaluate.py` per the Interfaces block.
- [ ] **Step 4: Run** `python3 -m unittest test_evaluate -v` → PASS.
- [ ] **Step 5: Commit** `git add evaluate.py test_evaluate.py && git commit -m "feat(decisionrag): evaluate against keyword and MiniLM baselines"`

### Task 6: CLI and README

**Files:**
- Create: `main.py`, `README.md`, `examples/nolan_corpus.json` (`{"documents": [{"id", "title", "passages": [{"id", "text"}]}]}`)
- Test: `test_main.py` (add `test_demo_fake_runs`: `main(["demo","--fake"]) == 0`)

**Interfaces:**
- Consumes: everything above.
- Produces: `main.main(argv: list[str] | None = None) -> int`.
  - `tag SUBJECT [--tags T …] [--declared C …] [--mode navigate|flat|separate] [--no-choice] [--fake]` — prints ranked tags (`id  p_bin  p_mul  source`) then the trace stages.
  - `demo [--fake]` — runs the paper's running example ("Who is the main character in Christopher Nolan's film from the year 2000?") over `examples/nolan_corpus.json`: three documents (Christopher Nolan, Memento (film), Gladiator (2000 film)), 3–4 short original passages each (Memento's includes its protagonist Leonard Shelby), loaded as a document → passage tree with config `k=2, tag_threshold=0.0, r=5, m=5`. Prints each stage's P(yes), what was kept and pruned, and the final passage ranking. It shows the engine on a non-taxonomy corpus with Jev.
  - `eval [--mode …|all] [--no-choice] [--fake] [--output out/report.json]` — `all` runs navigate, navigate `--no-choice`, flat, separate, plus baselines, and writes one report.
  - Judge selection: `--fake` → `FakeJudge()`; else requires `OPENROUTER_API_KEY` → `JevJudge(CachedTransport(HttpTransport(key), Path(".cache/jev")))`; missing key → stderr `"OPENROUTER_API_KEY is not set; export it or pass --fake"`, return 2.
- [ ] **Step 1: Write failing tests** in `test_main.py`: `test_missing_key_exits_2` (patch env without the key, `main(["tag","x"]) == 2`); `test_tag_fake_prints_tags` (capture stdout, return 0); `test_eval_fake_writes_report` (temp `--output`, `--mode all`, both files exist).
- [ ] **Step 2: Run** `python3 -m unittest test_main -v` → FAIL.
- [ ] **Step 3: Implement** `main.py`; write `README.md` (what it is, the paper link, the core idea in three lines, setup `export OPENROUTER_API_KEY=…`, the two commands, how to run tests, and that `.cache/` makes reruns free).
- [ ] **Step 4: Run** the whole suite `python3 -m unittest discover -v` → all PASS; then `python3 main.py eval --mode all --fake` → writes `out/report.json` and `out/report.md`.
- [ ] **Step 5: Commit** `git add main.py README.md test_main.py && git commit -m "feat(decisionrag): add CLI and README"`

### Task 7: Live run and results (needs the user's key)

**Files:**
- Create: `RESULTS.md`

- [ ] **Step 1:** Confirm `OPENROUTER_API_KEY` is set; if not, stop and ask the user to export it (do not proceed with `--fake`).
- [ ] **Step 2:** `python3 main.py tag "Private show with cosplay and latex" --tags bdsm lingerie` → ranked tags print; `usage` cost is a fraction of a cent.
- [ ] **Step 3:** `python3 main.py eval --mode all` → `out/report.md`. Rerun once and confirm the cache reports only hits (misses == 0; reported cost is nominal and stays nonzero).
- [ ] **Step 4:** Write `RESULTS.md`: the comparison table copied from `out/report.md`, per-slice notes (paraphrase, negation, benign_context), the `neutral-identity` case outcome, cost and latency, and the statement that defaults were fixed before the run. Any threshold sweep goes in a separate section labeled "tuned on test, not held out".
- [ ] **Step 5: Commit** `git add RESULTS.md && git commit -m "docs(decisionrag): record first live Jev results"`

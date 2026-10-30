# Auto Keyword Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, automatically built tree. It is built from `keywords.csv` with e5-base-v2 embeddings, two-level k-means, and cluster descriptions. It sits beside the existing taxonomy workflow, and its results are reported side by side with it.

**Architecture:** `tree_builder.py` builds the tree offline. Every heavy backend is injected: an embedder, a k-means function and a describer. It writes `trees/keyword_tree.json`. `keyword_corpus.py` loads that file into the existing `Node` tree and maps the selected keywords to concepts for scoring. The existing engine, the Jev judge and the taxonomy workflow are reused unchanged.

**Tech Stack:**
- Python 3.12 and `unittest`.
- **Build only**, imported lazily: `sentence-transformers` (e5-base-v2, which is already cached), `scikit-learn` and `numpy`.
- OpenAI Chat Completions, called over stdlib `urllib`.

**Spec:** `docs/superpowers/specs/2026-10-29-auto-keyword-tree-design.md`

All paths are relative to `AI-Agent-KnowledgeBase/ontology-use-case/Jev-RAG-DecisionModel/`. Run the tests from there with `python3 -m unittest discover`.

## Global Constraints

- **Purely additive.**
  - Existing commands, their defaults (`tag --tree taxonomy`) and the existing eval runs behave exactly as now.
  - Existing tests keep passing unchanged, except where a task says otherwise.
- **Dependencies stay out of the runtime and the tests.**
  - `torch`, `sentence_transformers`, `sklearn` and `numpy` are imported only inside the functions that use them. That means `E5Embedder`, `sklearn_kmeans` and centroid math done with numpy.
  - No test imports them, loads e5, or calls OpenAI. No test needs a key.
- **No taxonomy in tree building.**
  - Never read `taxonomy_v1.json`.
  - Never put a `concept_id` into any embedded text or describer prompt.
  - `concept_id` is stored on the keyword leaves only for scoring.
- **Fixed values:**
  - Input: all `keywords.csv` rows except `review_status == "excluded"`, which leaves 415.
  - Embedding: `intfloat/e5-base-v2` with the `"passage: "` prefix, normalized.
  - k-means: `random_state=0`, `n_init=10`.
  - Shape: 20 top clusters. A top cluster with more than 8 keywords is split into `ceil(n/5)` sub-clusters.
  - Describer prompt from spec B.5, `temperature=0`, at most 60 keywords sent, ordered by centrality.
  - Fallback description: `"Terms such as: " + ", ".join(first 8)`.
- **OpenAI:**
  - Model `DEFAULT_LLM_MODEL = "gpt-4.1-mini"`, overridable with `--llm-model`.
  - The model id actually used is recorded in the tree file.
  - The `.env` key is currently invalid, so the builder must degrade cleanly (see Task 2).
- **Keys:** never print, log, cache or commit any key.

## Review Focus

1. **An invalid or absent OpenAI key.** `build-tree` still succeeds: every node gets a fallback description, the tree file says so, and the build doesn't make ~105 failing calls. Tests in Task 2.
2. **A cluster with fewer than 2 keywords, or k greater than the number of points.** No crash: k is capped at the number of points. Test in Task 1.
3. **`tag --tree keywords` or `eval` with the tree file missing.** A clear message pointing to `build-tree` (exit 2), or a skip note in eval. Never a traceback. Tests in Task 5.
4. **A room whose selected keywords all have no concept.** The prediction row has an empty concept list, the unscorable count is greater than 0, and the evaluator accepts it. Test in Task 3.
5. **`keywords.csv` changed after the build.** A stale warning is printed and the run still goes ahead. Tests in Tasks 3 and 5.

---

### Task 1: Tree builder core (pure, injectable)

**Files:**
- Create: `tree_builder.py`, `test_tree_builder.py`

**Interfaces:**
- Produces:
  - **`load_keyword_rows(rows: Iterable[dict]) -> list[dict]`**: returns `[{"keyword", "concept_id"}]` with excluded rows dropped, in file order.
  - **`build_keyword_tree(rows, embed, kmeans, describe, *, clusters=20, leaf_size=5, split_over=8, seed=0) -> dict`**
    - `embed: Callable[[list[str]], list[list[float]]]`
    - `kmeans: Callable[[list[list[float]], int, int], list[int]]`, called as `(vectors, k, seed) -> labels`
    - `describe: Callable[[list[str]], str]`, which may raise
    - Returns `{"meta": {...}, "tree": node}`. A node is either `{"id", "description", "description_source": "llm"|"fallback", "children": [...]}` or a keyword leaf `{"id": "kw:<keyword>", "keyword", "concept_id"}`.
    - Root id `"__root__"`. Cluster ids are `"c<i>"` and `"c<i>.<j>"`.
    - `meta` has `counts` (`keywords`, `clusters`, `subclusters`, `fallback_descriptions`), `clusters`, `leaf_size`, `split_over` and `seed`.
  - **`fallback_description(keywords: list[str]) -> str`**.
  - **`write_tree(tree: dict, path: Path) -> None`** and **`read_tree(path: Path) -> dict`**: UTF-8 JSON, written atomically through a tmp file.
- Algorithm details the signatures leave open:
  - k is `min(k, len(points))`.
  - Each cluster's keywords are ordered by centrality, meaning squared Euclidean distance to the cluster's mean vector, computed in pure Python.
  - The describer gets the first 60 of those.
  - If `describe` raises any `Exception`, use the fallback and mark the node `description_source: "fallback"`.
  - Clusters are numbered in order of each cluster's first keyword's position in the file.

- [ ] **Step 0:** create branch `feat/jev-keyword-tree` from `refactor/jev-decisionrag-polish` (stacked on PR #109).
- [ ] **Step 1: Failing tests** in `test_tree_builder.py`. They use a fake embedder (each keyword maps to `[len(kw), ord(kw[0])]`), a fake k-means (`labels = [i % k for i in range(n)]`) and a fake describer (`"about " + kws[0]`).
  - `test_excluded_rows_dropped`: 3 rows, one excluded, gives 2.
  - `test_shape_and_leaf_rule`: 40 keywords with `clusters=4` give 4 top nodes with 10 keywords each, and each is split into 2 sub-clusters. With 12 keywords and `clusters=4`, each top cluster has 3 keywords (no more than 8), so its children are keyword leaves.
  - `test_every_keyword_appears_once`.
  - `test_k_capped_at_points`: 2 keywords and `clusters=20` give 2 top clusters (Review Focus 2).
  - `test_no_concept_text_in_embed_or_describe_inputs`: rows whose `concept_id` is `"SECRET_CONCEPT"`; the string never appears in any argument passed to `embed` or `describe`.
  - `test_describer_error_uses_fallback`: the describer raises, so the description is `"Terms such as: …"`, `description_source == "fallback"`, and `meta.counts.fallback_descriptions` equals the number of cluster nodes.
  - `test_write_read_roundtrip` (temp dir).
- [ ] **Step 2:** run `python3 -m unittest test_tree_builder`. It fails with ModuleNotFoundError.
- [ ] **Step 3:** implement `tree_builder.py` as specified.
- [ ] **Step 4:** run `test_tree_builder`, then the full suite. Everything passes.
- [ ] **Step 5:** commit `feat(keyword-tree): pure tree builder with injectable embed/kmeans/describe`.

### Task 2: Real backends (e5, k-means, OpenAI describer)

**Files:**
- Modify: `tree_builder.py`, `test_tree_builder.py`

**Interfaces:**
- Produces:
  - **`E5Embedder(model: str = "intfloat/e5-base-v2")`**: a callable `(texts) -> list[list[float]]`. It imports `sentence_transformers` lazily and calls `encode(["passage: " + t ...], normalize_embeddings=True)`.
  - **`sklearn_kmeans(vectors, k, seed) -> list[int]`**: a lazy `sklearn.cluster.KMeans(n_clusters=k, random_state=seed, n_init=10)`.
  - **`OpenAIDescriber(api_key: str, model: str = DEFAULT_LLM_MODEL, timeout: float = 60)`**: a callable `(keywords) -> str`.
    - POSTs to `https://api.openai.com/v1/chat/completions` with `temperature: 0` and the spec's B.5 prompt (adapted to "items"), with the keywords as a comma-separated list.
    - Returns the stripped message content.
    - It records `prompt_tokens`/`completion_tokens` totals and `model`.
    - **On HTTP 401/403 it sets `self.disabled = True` and raises `DescriberUnavailable`. Every later call raises `DescriberUnavailable` immediately, with no request** (Review Focus 1).
    - Other errors raise `RuntimeError`.
    - `__repr__` never includes the key.
  - **`DEFAULT_LLM_MODEL = "gpt-4.1-mini"`**, **`DESCRIBE_PROMPT`** (exact text from the spec).
- [ ] **Step 1: Failing tests**, with `urllib.request.urlopen` patched:
  - `test_openai_request_shape`: body `model`, `temperature == 0`, the prompt text present, and keywords in the user message.
  - `test_openai_401_disables_after_first_call`: two calls, `urlopen` called once, both raise `DescriberUnavailable`.
  - `test_build_with_unavailable_describer_completes`: `build_keyword_tree` with that describer gives a tree with every node on fallback.
  - `test_backends_import_lazily`: after `import tree_builder`, `"torch"`, `"sentence_transformers"` and `"sklearn"` are not in `sys.modules`. Check this in a subprocess, `python3 -c "import tree_builder, sys; print(...)"`.
- [ ] **Step 2:** run them. They fail.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run `test_tree_builder` and the full suite. Everything passes.
- [ ] **Step 5:** commit `feat(keyword-tree): e5 embedder, k-means and OpenAI describer backends`.

### Task 3: Keyword-tree adapter and concept scoring

**Files:**
- Create: `keyword_corpus.py`, `test_keyword_corpus.py`
- Modify: `taxonomy_corpus.py` (`tag_room` flat/separate uses `config.returnable` instead of `is_returnable`. This is the same behavior, because `taxonomy_config` sets `returnable=is_returnable`.)

**Interfaces:**
- Consumes: `read_tree` (Task 1), `tag_room`, `room_query` and `taxonomy_config`'s question wording from `taxonomy_corpus`.
- Produces:
  - **`KEYWORD_TREE_PATH = HERE / "trees" / "keyword_tree.json"`**.
  - **`KEYWORDS_CSV = ONTOLOGY_DIR / "keywords.csv"`**.
  - **`load_keyword_tree(path=KEYWORD_TREE_PATH) -> tuple[dict, Node]`**: returns `(meta, root)`.
    - Cluster `Node.description` is the description.
    - Leaf `Node.description` is the keyword, with `payload={"keyword", "concept_id"}`.
    - Raises `FileNotFoundError` if the file is missing.
  - **`is_stale(meta, csv_path=KEYWORDS_CSV) -> bool`**: the sha256 of the csv bytes differs from `meta["keywords_sha256"]`.
  - **`keyword_config(**overrides) -> NavigationConfig`**: returnable is leaves only; the questions are the same as `taxonomy_config`.
  - **`to_concepts(result: Result, root: Node) -> tuple[list[str], int]`**: returns the concept ids in rank order (self-declared first, deduplicated keeping the first) and the count of selected keywords with an empty `concept_id`.
- Also, `build_keyword_tree` (Task 1) must store `meta["keywords_sha256"]`. Pass it in via a new keyword argument, `csv_sha256: str | None = None`, recorded as-is. Add that to `tree_builder.py` here.
- [ ] **Step 1: Failing tests**, using a small hand-written tree file written to a temp dir:
  - `test_load_tree_nodes_and_payload`
  - `test_missing_file_raises`
  - `test_is_stale` (same bytes give False; changed bytes give True)
  - `test_to_concepts_maps_dedups_and_counts_unscorable`: the keywords `[a(c1), b(c1), x(""), c(c2)]` give `(["c1", "c2"], 1)`.
  - `test_all_unscorable_gives_empty_list` (Review Focus 4)
  - `test_tag_room_flat_uses_config_returnable`: keyword tree, flat mode, FakeJudge; only `kw:` ids are returned.
  - `test_taxonomy_flat_unchanged`: the existing taxonomy flat result ids are equal before and after (compared against `is_returnable` units).
- [ ] **Step 2:** they fail. **Step 3:** implement. **Step 4:** the module tests and the full suite pass.
- [ ] **Step 5:** commit `feat(keyword-tree): keyword tree adapter and concept mapping`.

### Task 4: Evaluation runs and the comparison report

**Files:**
- Modify: `evaluate.py`, `test_evaluate.py`

**Interfaces:**
- Consumes: Task 3's adapter.
- Produces:
  - **`run_keyword_tree(judge, config, mode="navigate", name=None, tree_path=KEYWORD_TREE_PATH) -> dict`**: has the same keys as `run_decisionrag`, plus `"unscorable_hits": int`, `"rooms_with_unscorable": int` and `"tree": "keywords"`.
    - Each row's `concept_ids` come from `to_concepts`.
    - Each row also has `"keywords": [ranked keyword ids]` and `"unscorable": int`.
    - The default name is `f"decisionrag-keyword-{mode}"`, and `mode="navigate"` gives `"decisionrag-keyword-tree-navigate"`.
  - **`write_report`** gains a **`hier. micro-F1`** column, from `report["hierarchical"]["micro_f1"]`, right after micro-F1, for every run.
    - When any run has `unscorable_hits`, it appends the line `"Unscorable keyword hits (no concept): <name>=<n> in <rooms> rooms; …"`.
- [ ] **Step 1: Failing tests:**
  - `test_run_keyword_tree_fake`: a hand-written tree whose leaves cover concepts used in the cases, with FakeJudge. Gives `count == 28` and integer `unscorable_hits`.
  - `test_report_has_hierarchical_column_and_unscorable_note`.
- [ ] **Step 2:** they fail. **Step 3:** implement. **Step 4:** the tests and the full suite pass.
- [ ] **Step 5:** commit `feat(keyword-tree): eval runs, hierarchical F1 column, unscorable note`.

### Task 5: CLI, per-command keys, README

**Files:**
- Modify: `main.py`, `test_main.py`, `README.md`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - **`build-tree [--clusters 20] [--leaf-size 5] [--out PATH] [--llm-model ID] [--fake]`**
    - Real mode uses `E5Embedder()`, `sklearn_kmeans`, and `OpenAIDescriber(os.environ["OPENAI_API_KEY"], model)` when the key is set. Otherwise it uses the fallback describer directly, with a printed note.
    - `--fake` uses Task 1's fakes and defaults `--out` to `out/keyword_tree.fake.json`.
    - It reads `KEYWORDS_CSV` and records its sha256, the embed model, the LLM model (or `"none"`) and the build date in `meta`.
    - It prints cluster counts, the number of fallback descriptions, and OpenAI token totals.
    - It needs **no** Jev key.
  - **`tag ... --tree taxonomy|keywords`** (default `taxonomy`).
    - With `keywords`, it prints each keyword with its concept in brackets.
    - If the tree file is missing, it prints `keyword tree not found at <path>; run: python main.py build-tree` to stderr and exits 2.
    - If the tree is stale, it prints the warning `keyword tree is stale (keywords.csv changed); run: python main.py build-tree` and continues.
  - **`eval --mode all`** appends `decisionrag-keyword-tree-navigate` and `decisionrag-keyword-flat` (a judge with the default `max_per_request=50`) **when the tree file exists**. Otherwise it prints `skipping keyword-tree runs: <path> not found`. A stale tree also warns.
  - **Key checks move per command.** `build-tree` never builds a Jev factory. `tag`, `demo` and `eval` keep today's behavior and messages.
- [ ] **Step 1: Failing tests** (hermetic, `env_paths=()`, patched `make_judge_factory` where needed):
  - `test_build_tree_fake_writes_file` (temp `--out`)
  - `test_build_tree_needs_no_jev_key` (clean env, `--fake`, exit 0)
  - `test_tag_keywords_missing_tree_exits_2` (patch `KEYWORD_TREE_PATH` to a missing path)
  - `test_tag_keywords_stale_warns` (temp tree with a wrong sha)
  - `test_eval_all_skips_keyword_runs_without_tree`
  - `test_existing_commands_unchanged` (the existing tag, demo and eval tests still pass untouched)
- [ ] **Step 2:** they fail. **Step 3:** implement. Then add a README section, "Auto-built keyword tree": what it is, `build-tree`, `--tree keywords`, that `OPENAI_API_KEY` is optional with keyword-list fallback, the extra build-only dependencies, and the stale warning. Also add a step to the step-by-step list.
- [ ] **Step 4:** the full suite passes. `python3 main.py build-tree --fake`, `python3 main.py eval --mode all --fake` and `python3 main.py demo --fake` run.
- [ ] **Step 5:** commit `feat(keyword-tree): build-tree command, --tree keywords, eval integration`.

### Task 6: Live build, live eval, comparison results (controller)

- [ ] **Step 1:** run `python3 main.py build-tree` for real: e5 locally, k-means, and OpenAI if the key works, otherwise the fallback. Check the counts: 415 keywords, 20 top clusters. Commit `trees/keyword_tree.json`.
- [ ] **Step 2:** run `python3 main.py tag "Private show with cosplay and latex" --tags bdsm lingerie --tree keywords` live. Check that the output is sensible.
- [ ] **Step 3:** run `python3 main.py eval --mode all` live. Most of the existing runs are served from the cache, and only the new runs cost money.
- [ ] **Step 4:** add a "Comparison: hand-built taxonomy tree vs auto keyword tree" section to `RESULTS.md`, keeping the first section as is. Include:
  - the table from the spec's goal 4
  - per-slice notes
  - the unscorable-hit counts
  - whether the descriptions came from an LLM or the fallback, and that this caps the auto tree's quality
  - cost and latency
- [ ] **Step 5:** commit `docs(keyword-tree): live comparison results`, push, and open a PR stacked on #109.

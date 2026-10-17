# JevBench Snapshot: OpenAI Template, No Old-Model Figures

## Goal

Finish moving Jev-Evaluation to OpenAI models. After #77–#80, the only old-model references left were the archived 2026-09-26 benchmark: `figures/jevbench-2026-09-26/` (`snapshot.json` and five SVG charts) and the two scripts that build it. Point those scripts at Jev plus the OpenAI lineup, so the next real run produces OpenAI figures, and remove the old figures.

## Decision (user's choice)

**Template, no fake scores.** The archived figures show measured scores for DeepSeek, Doubao, GLM and Kimi. They are deleted rather than relabelled, because relabelling would show scores the OpenAI models never earned. They stay in git history. No benchmark is run as part of this change.

## Design

### `benchmark/scripts/prepare_jevbench_snapshot.py`

- `MODEL_IDS = ["jev", *OPENAI_MODELS]`, imported from `benchmark/llm_eval/adapters`. That is the same table the runner uses, so a run's directory names (`runs/notebook-demo/real/<id>/`) line up. The display name is `Jev` for Jev and the model id for the OpenAI models.
- `--date YYYY-MM-DD` (default: today) chooses the output `figures/jevbench-<date>/snapshot.json`, and it is recorded as `metadata.snapshot_date`. The date was previously hard-coded.
- Models missing from the archive are skipped with a printed `skip <id>` line, so a partial run still produces a snapshot. `metadata.models_order` lists only the models present. With none present, the script exits with an error.

### `benchmark/scripts/render_jevbench_snapshot.py`

- `--snapshot PATH` (default: the newest `figures/jevbench-*/snapshot.json`). The SVGs are written next to it.
- The dashboard title and date come from the snapshot (`{n}-provider snapshot`, `snapshot_date`). They were hard-coded as "five-provider" and "26 Sep 2026".
- The layout scales with the number of models, using two helpers:
  - `row_positions(start, count, span, max_step)` spaces rows evenly and tightens them when there are more rows, instead of overflowing.
  - `column_width(count, available, gap, max_width)` narrows the columns so they fit.
- These replace the fixed `range(5)` and `gap * 5` geometry. With five models both helpers return the original values, so re-rendering the archived snapshot reproduced four SVGs byte for byte. The fifth differed only in the intended title wording.
- `MODEL_COLORS` keys now cover the new lineup. The dict is not read anywhere; it is kept as the palette reference.

## Non-goals

- Running the benchmark or producing new figures.
- Changing metrics, the composite formula, or what the charts show.

## Verification

- `benchmark/tests/test_snapshot_scripts.py` (5 tests) builds a synthetic results archive in `tmp_path` and checks that:
  - the lineup is Jev plus `OPENAI_MODELS`
  - `prepare` writes a dated snapshot and skips missing models
  - `render` fits six models inside every canvas and shows the count and date
  - the layout helpers reproduce the five-model geometry
- All 5 fail against the previous scripts.
- Manual checks: the archived five-model snapshot re-renders identically apart from the title; a six-model render keeps every rect and label inside its canvas; `prepare` ran on a zip of stub-API run results and wrote the five OpenAI models, skipping Jev.

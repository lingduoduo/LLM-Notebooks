# Application code investigation and optimization — 2026-10-17

Reviewed the Python game engines and spectator servers, browser agent and model adapter, smart-home proxy, maze and Doom environments, Mario attempt memory, Node game servers and comparison helpers, and React entry point. Changes target demonstrated hot paths and reproduced correctness failures. No live model calls were made.

Baseline: `e1adff3784b4037edc04fd9cab074ff02cd2188f`. Tests used an isolated Python 3.12 environment with pytest, httpx[http2], and numpy. Application runtime dependencies were not changed.

## Implemented improvements

| Component | Finding and change |
|---|---|
| Sudoku candidates | `_candidates` rebuilt the same row/column/box exclusions separately for all nine digits. Compute them once. Reuse the selected cell's candidates when rendering digit criteria. |
| Sudoku generation | RNG stored on a shared function attribute let overlapping generation calls change seeded puzzles. Keep the RNG local to each generation call. A reentrant generation regression reproduces the original failure deterministically. |
| Sudoku state | Zero-hole puzzles were returned as live states even though already solved, making the runner fail and the validator reject its own generator output. Return an immediate win with zero steps. |
| Sudoku validation | Box indexing checked the top-left box nine times. Check all nine boxes; regression swaps rows across the lower bands while preserving valid rows and columns. |
| Dou Dizhu policy | Following could pass to a teammate despite having an immediate winning play. Check winning moves first. Leading minimized a positive card-count penalty, favoring singles contrary to the published instructions; prioritize the largest non-explosive combination, then lower rank. |
| Dou Dizhu records/server | Reuse one generated legal-move list to render a record, and reuse its public request in the server. Previously these paths generated and rendered the same moves repeatedly. Judges still receive only public request fields, never gold labels. |
| Blackjack and Dou Dizhu servers | Non-dictionary judge replies crashed game threads, and a list-valued move ID crashed dictionary lookup. Validate replies and apply the existing legal fallback. Dou Dizhu now reports the fallback it actually executed instead of an invalid ID. |
| Mario memory | Stored prompts referenced caller-owned nested objects, allowing later changes to rewrite recorded history. Copy prompts at recording time. Preserve legitimate zero outcome coordinates instead of replacing them with earlier coordinates. |
| Browser text helper | An empty provider choices list escaped validation as IndexError. Reject it with the same clear ValueError used for other malformed provider responses. |
| Minesweeper comparison | Out-of-range columns could alias valid board cells (`r1c15` became cell 14). Validate row and column bounds before converting coordinates. |
| Shared launcher | Probe services concurrently while retaining card order; explicitly close HTTP responses; shell-quote the Python executable when expanding runnable commands. |
| Test collection | Load each app's `experiment.py` explicitly to avoid cross-app module-name collisions. Make browser tests importable when pytest runs from the chapter/repository root. |

## Measured performance

| Workload | Before | After | Result |
|---|---:|---:|---|
| Generate 10 Sudoku puzzles, seeds 0–9, 50 requested holes | 0.9213 s | 0.1475 s | About 6.2× faster |
| Generate 1,000 Dou Dizhu records, seeds 0–99 repeated 10 times; median of 5 runs | 0.2052 s | 0.1720 s | About 16% less time |
| Render menu with seven simulated 150 ms service probes | 1.084 s | 0.156 s | About 6.9× faster for this simulated workload |

Sudoku profiling originally attributed 1.558 of 1.639 profiled seconds to 379,953 repeated peer scans when generating five 50-hole puzzles. All ten complete Sudoku states in the timing workload, including grids, solutions, masks and RNG state, match the baseline exactly. Dou Dizhu public requests match the baseline across 1,000 observed states, and reference distributions remain normalized.

These are local CPU/latency measurements, not model or gameplay-success benchmarks. Dou Dizhu reference **labels intentionally change** with the policy corrections; frozen experimental artifacts were not regenerated. Mario snapshots now require a defensive copy, trading that bounded work for accurate history.

## Verification

New correctness regressions were observed failing before their corresponding fixes.

| Check | Result |
|---|---|
| Combined launcher, Blackjack, Dou Dizhu, Sudoku, smart-home, browser and Mario regression suites | 123 passed |
| Maze `test_unified_grid_envs.py` and `test_scaled_games_fixes.py` | 18 passed, 150 subtests passed |
| Doom/demo/filter tests | 30 passed, 1 skipped, 4 subtests passed |
| Node game-engine, speech, server-request and Minesweeper comparison tests | 23 passed |
| React `npm run build` | Passed TypeScript and Vite production build |
| Full chapter pytest with continued collection | 191 passed, 4 failed, 2 skipped, 11 collection errors; 205 subtests passed |
| `git diff --check` | Passed |

Representative commands, using the isolated environment's Python:

```bash
python -m pytest Jev_Application/test_start.py \
  Jev_Application/app/blackjack/test_blackjack_game.py \
  Jev_Application/app/doudizhu/test_doudizhu_game.py \
  Jev_Application/app/sudoku/test_jev_sudoku.py \
  Jev_Application/app/smart-home/test_serve_smart_home.py \
  Jev_Application/app/browser-use/tests/test_fixes.py \
  Jev_Application/app/typesafe-mario-repro/test_harness_fixes.py -q
node --test Jev_Application/app/jev-games/scripts/*.test.mjs \
  Jev_Application/app/jev-games/games/werewolf/*.test.mjs
python -m pytest Jev_Application -q --continue-on-collection-errors
# From app/jev-games/apps/web/:
npm run build
```

## Remaining full-suite limitations

Four runtime failures require training modules absent from this folder:

- `test_composed_maze.py::LocalObservationTests::test_boundaries_and_local_truth_match_geometric_neighbors`
- `test_scaled_maze.py::IdentityTests::test_cross_split_map_is_rejected_by_training_loader`
- `test_scaled_maze.py::ObservationAndTargetTests::test_forced_terminal_and_choice_schema`
- `test_scaled_maze.py::ObservationAndTargetTests::test_mixed_records_full_curriculum_train_contract`

The eleven collection errors also involve absent training/evaluation modules:

| Test file | Missing module |
|---|---|
| maze/scripts/test_build_local_maze_data.py | predict_toy_decisions |
| maze/scripts/test_game_outcomes.py | train_pipeline_decisions |
| maze/scripts/test_model_edges_maze.py | evaluate_native_qwen_maze |
| maze/scripts/test_scaled_pipeline.py | train_pipeline_decisions |
| predict_position/scripts/test_complete_sonic_evaluation.py | train_unified_games |
| predict_position/scripts/test_native_qwen_shooting.py | evaluate_native_qwen_navigation |
| predict_position/scripts/test_prepare_sonic_supervision.py | train_pipeline_decisions |
| predict_position/scripts/test_run_sonic_supervision.py | train_unified_games |
| predict_position/scripts/test_sonic_dev_epsilon.py | appo_basic_data |
| predict_position/scripts/test_sonic_predict_data.py | appo_basic_data |
| predict_position/scripts/test_summarize_sonic_supervision.py | summarize_unified_games |

These missing training dependencies were already identified in TESTING.md; no replacement stubs were added to production or failing tests skipped to make the full-suite result appear green. Live browser operation, real API latency/cost, and full NES/ViZDoom gameplay were not validated by this offline pass.

The launcher still stops services by port rather than verified process ownership. That preexisting behavior was inspected but not exercised or changed.

Independent code review examined the 15 modified tracked files and reported no concrete regressions or blockers. Test results above were run by the implementing agent using the isolated environment.

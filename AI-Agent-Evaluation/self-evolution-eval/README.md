# Experiment 9-9: Evaluating Whether an Agent Continues to Evolve

This experiment expands evaluation from success on a single task to performance across a long-running task stream. Tasks do not simply repeat. They progress through four phases: learning exposes shared patterns; transfer changes the wording and environment; rule changes require revising existing capabilities; and retention retests unchanged capabilities and currently valid rules.

```bash
# Reference agents validate the harness; they do not count as live experiment acceptance.
python -m pytest -q test_longitudinal.py test_campaign_statistics.py
python demo.py --profile all --output output/reference-report.json

# Live acceptance: 3 real-model arms x 3 seeds x 14 sequential tasks = 126 API calls.
python run_experiment_9_9.py \
  --provider ark --model doubao-seed-1-6-250615 \
  --seeds 8601,8602,8603 --workers 6
```

`dataset.json` contains three task families: refunds, identity verification, and baggage policy. In the third phase, the baggage allowance changes from 20 kg to 23 kg. An agent that only appends knowledge without retiring old rules therefore continues to fail in the change and retention phases. The reference-agent path is entirely offline. The live acceptance path requires an API key, and a real model makes every task decision in every arm.

The three real-model arms share the same model, task order, seed schedule, and prompting protocol. They differ only in the lifecycle of memory stored outside the model:

- `static` never persists feedback.
- `append_only` stores every observation and conflict but always keeps the first rule version active.
- `evolving` stores versions and provenance, replaces old rules with higher versions, and retains `superseded` audit records.

```bash
# From the repository root, use the shared Chapter 8 environment.
uv sync --locked --python 3.12 --extra ch8
# Apple Silicon requires macOS 14+ for the bitsandbytes wheel in the lockfile.
# On older macOS versions, use the standalone compatibility setup below.

# Activate the environment before changing directories:
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

# If uv is unavailable, use pip as a fallback:
# python -m pip install -e ".[ch8]"

cd AI-Agent-Evaluation/self-evolution-eval

# The standalone compatibility setup remains supported during migration:
# python -m pip install -r requirements.txt

export OPENAI_API_KEY=your_api_key_here
python demo.py --profile llm --model gpt-5.6 --output output/llm-report.json
```

The harness exposes the learning signal only after the model returns and its current action is recorded. Requests sent to the model contain only task inputs and previously activated memories, excluding `expected_action` and `learning_signal`. Raw requests and responses, response IDs, seeds, timestamps, token counts, latency, and hashes are written to `validation/<run>/evidence.json`. The most recent canonical evidence is stored in `validation/latest.json`. Credential values are never recorded in the evidence.

`demo.py` provides three controlled reference agents to check that the metrics distinguish the intended behaviors:

- `evolving` retains experience and replaces old rules with higher versions.
- `append_only` learns the first rule version but cannot update or retire it.
- `static` persists no production feedback.

These are reference implementations, not real models. They check whether the evaluation framework can distinguish three long-term behaviors. To replace `ReferenceAgent` with your own agent, implement `act(task)`, `observe(task)`, `profile`, and `storage_bytes`.

## Results from Repeated Real-Model Experiments

The canonical run stored in the repository used Ark `doubao-seed-1-6-250615` with seeds 8601, 8602, and 8603. The core rates were identical across all three repetitions:

| Arm | Transfer accuracy | Adaptation recovery score | Rule replacement accuracy | Obsolete rule reference rate | Retention rate | Unchanged capability retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `append_only` | 1.000 | 0.000 | 0.000 | 1.000 | 0.667 | 1.000 |
| `evolving` | 1.000 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 |

`evolving` recovered on the next task after receiving the first 23 kg rule signal and continued using 23 kg during retention. `append_only` transferred well but continued citing 20 kg in every subsequent rule-replacement check. This is the distinction the experiment aims to measure: retaining knowledge versus updating it. Across 126 calls, the run consumed 48,318 input tokens, 35,222 output/reasoning tokens, and 83,540 total tokens. The provider returned no monetary cost field, so the evidence reports measured token counts, latency, and storage without estimating dollar costs.

## Reported Metrics

`LongitudinalEvaluator` reports per-phase accuracy, learning curves, transfer accuracy, recovery speed after rule changes, rule replacement accuracy, obsolete rule reference rate, unchanged capability retention, current-rule retention, negative transfer rate, safety rubric pass rate, and token, time, and storage costs. It also separately reports modification proposal validity, artifact activation, and memory adherence rates. Repeated experiments provide each metric's mean, sample standard deviation, and 95% t interval, along with paired `evolving-static` and `evolving-append_only` differences for matching seeds.

Recovery speed measures how many additional tasks are needed to answer correctly after receiving the first new-rule signal. Negative transfer counts cases where an agent uses existing experience and consequently answers incorrectly. Retention is measured only against currently valid rules in the final phase, so continued use of a retired policy is not mistaken for good memory.

The experiment deliberately avoids compressing all metrics into one overall score. An agent may transfer well but fail to update old knowledge, or retain knowledge well while taking shortcuts that violate rules. Interpreting continual evolution requires visibility into adaptability, retention, efficiency, and safety together.

## Files

| File | Purpose |
| --- | --- |
| `dataset.json` | Four-phase sequential task stream and environment feedback |
| `agent.py` | Three reference behaviors and the static, append-only, and evolving real-model arms |
| `harness.py` | Longitudinal execution, per-phase statistics, cost tracking, and safety evaluation |
| `demo.py` | Command-line comparison experiment |
| `run_experiment_9_9.py` | Repeated, seeded, three-arm real-model experiments with statistics and evidence generation |
| `test_longitudinal.py` | Tests for transfer, rule updates, retention, and four-phase completeness |
| `test_campaign_statistics.py` | Tests for repeated-run means, sample standard deviations, and t intervals |

The earlier four-layer evaluation of tool discovery, creation, and reuse is no longer the chapter's main experiment. Tool creation can still serve as one mechanism for updates within a continual-evolution loop, but it alone cannot establish that an agent adapts to changes and avoids forgetting over long-running tasks.

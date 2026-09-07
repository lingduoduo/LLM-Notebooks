# Experiment 9-1: A Three-Layer Trajectory Verifier for Customer-Service Agents

This experiment accompanies the section "Obtaining Learning Signals from Execution Trajectories." Instead of treating user satisfaction or a single overall score as the learning signal, it checks environment outcomes, execution processes, and language quality in sequence, preserving evidence turn numbers for every failing dimension.

`verifier.py` implements three layers: the result layer reads the final order state; the process layer checks business rules, privacy, factual grounding, and promise-action consistency; and the quality layer evaluates open-ended criteria using a rubric for expression quality and compliant flexibility. The example defaults to the deterministic `HeuristicQualityJudge`, so no API key is required. A real LLM implementation follows the same `QualityJudge` interface, while the two lower layers continue to rely on environment ground truth and programmatic rules.

`sample_trajectories.json` contains four types of trajectories with expert labels: a normal refund, a false promise, an unauthorized disclosure, and an excessive refusal. `calibration.py` reports violation-detection precision, recall, and label agreement for each dimension. `demo.py` also compares a single overall score with multidimensional diagnostics backed by evidence.

## Code map

- **Run first:** python demo.py (deterministic HeuristicQualityJudge, no API key).
- **Start here:** verifier.py composes the result, process and quality layers.
- **Core behavior:** customer_service_env.py::run_case supplies environment truth; calibration.py compares dimensions with expert labels.
- **State / protocol:** sample_trajectories.json, structured verdict schema and evidence turns.
- **Verifier:** test_verifier.py plus calibration precision/recall; LLM quality judging never replaces the first two code gates.
- **Experiment variable:** single scalar score versus dimensioned verdict with evidence/confidence.
- **Skip on first pass:** provider client and demo formatting.

Run the deterministic example:

```bash
python demo.py
python -m unittest -v test_verifier.py
```

The commands above use deterministic calibration. To call a real LLM to evaluate expression quality and compliant flexibility:

```bash
# From the repository root, use the shared Chapter 8 environment.
uv sync --locked --python 3.12 --extra ch8
# Apple Silicon requires macOS 14+ for the bitsandbytes wheel in the lockfile.
# For older macOS versions, use the standalone compatibility setup below.

# Activate the environment before changing directories:
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

# If uv is unavailable, use pip as a fallback:
# python -m pip install -e ".[ch8]"

cd AI-Agent-Evaluation/trajectory-verifier

# The standalone compatibility setup remains supported during migration:
# python -m pip install -r requirements.txt

cp env.example .env
export OPENAI_API_KEY=your_api_key_here
python demo.py --judge llm --model gpt-5.6
```

Live mode uses the OpenAI Responses API and requires the model to return per-dimension verdicts, evidence turn numbers, and confidence using the same schema. Code still evaluates the environment-result and process-rule layers. This command incurs API charges, and outputs may vary by model version. Continue checking each dimension against expert labels rather than relying only on the overall score.

Production systems should expand the expert calibration set and route low-confidence or high-risk trajectories to a second verifier or human review. The sample `quality_facts` fields explicitly represent LLM judgments for the offline experiment; they do not imply that production systems can obtain these fields in advance.

The files under `validation/` are English translations of historical validation artifacts, including recorded model responses and reasoning text. Their recorded scores, labels, and run metadata are retained; they are not new evaluations of the translated text. Source anchors are URL-encoded to preserve the original reference. The claim matcher retains its existing multilingual support through Unicode escapes.

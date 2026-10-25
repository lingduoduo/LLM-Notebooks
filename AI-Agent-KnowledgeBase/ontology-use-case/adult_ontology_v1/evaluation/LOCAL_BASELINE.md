# Cached local semantic baseline

This experiment compares the existing keyword matcher with a **zero-shot
embedding-similarity baseline**. Both use the same 28 original held-out cases and
taxonomy 1.3.0. The taxonomy, vocabulary, gold labels and policy routing remain
unchanged. No third-party evaluation text or new weights were downloaded.

## Model and offline execution

The default model is the already-cached
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its model card describes
mean pooling over contextual token embeddings with the attention mask, followed
by L2 normalization. This implementation follows that recipe with Transformers
on CPU; it does not use the sentence-transformers package.

`local_encoder.py` resolves the pinned snapshot with `local_files_only=True` and
`token=False`, then loads the tokenizer and safetensors from that local folder
with `local_files_only=True` and `trust_remote_code=False`.
[Transformers offline loading](https://huggingface.co/docs/transformers/installation#offline-mode)
is explicit at both resolution and loading. The model run was also checked with
a Python audit hook blocking socket connections: **zero connection attempts**.

Actual inference needs optional `torch`, `transformers` and `huggingface_hub`
packages plus complete cached model files. Nothing is installed automatically.
The ordinary evaluator, command help and unit tests remain standard-library-only.
Missing weights or dependencies produce an error; the command never falls back
to downloads. Use an existing environment containing the dependencies:

```bash
# From the ontology directory, using your model-enabled Python:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
python -m evaluation.compare_semantic --output-dir /tmp/room-semantic-comparison

# Alternatively, supply a complete local MiniLM-compatible mean-pooling folder:
python -m evaluation.compare_semantic --model-path /path/to/local/model --output-dir /tmp/room-semantic-comparison
```

Explicit local folders are identified by file hashes rather than an asserted Hub
revision. They must include config.json, tokenizer.json and model.safetensors;
configuration/special-token/vocabulary files present in the folder are also
fingerprinted. Do not assume arbitrary transformer models use this pooling or
query format. Inputs are truncated to 256 tokens per field.

## Prediction rule fixed before evaluation

- Build one prototype per non-identity L1–L3 concept using **only its label and
  parent path**. No benchmark examples, expected labels or keyword terms enter
  prototype construction. Custom prototype injection is available in the Python
  API; the recorded run uses generated metadata-only prototypes.
- Embed prototypes once. Embed the subject and up to five nonempty tags
  separately, preserving raw input text for inference.
- Score each concept by its maximum cosine similarity across fields. Retain
  scores of at least **0.5**, remove qualified ancestors of qualified descendants,
  then keep at most **three semantic candidates**, ordered by score and stable ID.
- Abstain when nothing reaches the threshold; do not force a nearest label.
- Never infer identity labels from text. Add explicit self-declared IDENTITY
  concepts separately; they do not consume the semantic-candidate limit or get
  fabricated similarity scores.
- Retain the best matching raw field as evidence. Similarity does not establish a
  precise supporting span, semantic entailment, or calibrated probability.
  Every semantic candidate requires context review; abstentions and standalone
  identity declarations do not get a model-review flag.

The defaults were recorded before the held-out run. No cutoff search, model
selection, prompt adaptation or gold-label tuning was performed after seeing
scores. The CLI exposes `--threshold` and `--max-labels` for future experiments;
choose such settings on a separate validation set, not on these test cases.

## Outputs and reproducibility

The command writes:

- `semantic_comparison.json`: both detailed evaluator reports, selected settings,
  package versions, model revision/artifact hashes, prototype/taxonomy hashes and
  benchmark/vocabulary hashes.
- `semantic_predictions.jsonl`: actual model predictions, candidate similarities,
  raw-field evidence, review flags, case IDs and taxonomy version.
- `semantic_prototypes.json`: exact texts used to embed concepts.

The checked-in [summary](local_semantic_summary.json) and
[predictions](local_semantic_predictions.jsonl) preserve the measured run without
bundling model weights or machine-specific cache paths. The recorded predictions
can be scored by the standard-library evaluator:

```bash
python -m evaluation.semantic_evaluation --predictions evaluation/local_semantic_predictions.jsonl
```

## Observed results

| Metric | Keyword matcher | MiniLM prototype baseline |
| --- | ---: | ---: |
| Concept precision | 0.667 | 0.429 |
| Concept recall | 0.692 | 0.346 |
| Concept micro-F1 | 0.679 | 0.383 |
| Exact concept-set matches | 15/28 | 10/28 |
| Paraphrase-slice micro-F1 | 0.000 | 0.400 |
| Taxonomy-negative cases with any candidate label | 8/10 | 3/10 |
| Extra predicted concept labels across all cases | 9 | 12 |
| Missed gold concept labels across all cases | 8 | 17 |

MiniLM recovers some paraphrases, but this simple prototype setup has lower
precision, recall and exact-match accuracy overall. Fewer negative *cases* with
candidates does not mean fewer extra *labels* overall. Review flags do not erase
errors from scoring. The generic embedding model is not trained to resolve room
negation, advertised-versus-discussed features or safety-policy outcomes.

These are uncalibrated measurements on a tiny synthetic challenge fixture.
They establish neither production accuracy nor a general advantage for either
approach. The keyword implementation remains the default classifier; the local
model is an explicit experimental comparison. No production-classification
behavior changes automatically after this experiment.

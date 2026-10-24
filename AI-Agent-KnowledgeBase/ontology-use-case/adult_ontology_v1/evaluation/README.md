# Dataset-inspired semantic evaluation

Borrow annotation and evaluation ideas from public Hugging Face dataset cards,
not their text. The bundled 28 cases are original, non-graphic synthetic examples
for taxonomy **1.3.0**. No external rows are downloaded or redistributed, no
access conditions are accepted. The default evaluator uses no model or API; an
optional [cached local-model comparison](LOCAL_BASELINE.md) runs explicitly.

## Sources and what transfers

The dated [source registry](dataset_sources.json) records card URLs, displayed
card revisions, declared licenses, access observations, quality questions and
mapping limitations. These are source claims, not independently audited rights
or quality findings.

| Public card | Borrowed idea | Reuse/quality observation from the card |
| --- | --- | --- |
| [X-Sensitive](https://huggingface.co/datasets/cardiffnlp/x_sensitive/blob/main/README.md) | Multiple labels, evidence highlights and fixed splits | CC BY 2.0 metadata appears alongside a noncommercial gated-access condition; resolve the terms before data reuse. Social posts differ from room descriptions. |
| [NSFW-Safe](https://huggingface.co/datasets/eliasalbouzidi/NSFW-Safe-Dataset/blob/main/README.md) | Retain original/preprocessed variants and record transformations | Declares Apache 2.0; describes scraped/aggregated scene text. Its preprocessing removes numbers and punctuation, which can matter here. Upstream provenance still needs review. |
| [Valurank Adult Content](https://huggingface.co/datasets/valurank/Adult-content-dataset/blob/main/README.md) | Explicit text-label pairs and negative examples | Lists license `other` and limited annotation/source detail. Specific reuse terms were not established. |
| [DaijobuAI Moderation](https://huggingface.co/datasets/DaijobuAI/moderation_dataset/blob/main/README.md) | Define labels and distinguish upstream tasks | Declares MIT, lists derived sources and an intended-use statement; moderation/fake-news labels are not room concepts. |

A binary Adult/NSFW label or broad `sex` label cannot become BODY, ACT, KINK or
any specific L3 concept automatically. Likewise, format, commerce, generic
language and neutral identity are not safety-policy outcomes. Importing external
data later requires inspecting source terms/provenance, pinning data revisions,
reviewing labels and independently annotating room concepts. This tool does not
import, accept access terms for, or approve any dataset.

## Gold and quality boundaries

`semantic_cases.json` targets **advertised room features**. A reference in a
negated offering, educational discussion or benign alternative sense is not an
asserted feature. Identity labels require explicit self-declaration. Profanity
can belong to GENERIC without implying content. Gold was authored separately
from keyword predictions; it must not be generated from the matcher or adjusted
merely to improve its scores.

Slices include exact matches, multiple branches, paraphrases, negation,
educational references, benign senses, word boundaries, generic language,
neutral identity, explicit declarations, Unicode and tag evidence. All bundled
cases are held-out `test` cases. `group_id` keeps related examples together;
validation rejects groups or normalized identical inputs crossing splits.
This detects declared-group and exact-normalized leakage, not every semantic
near-duplicate. Future evaluation should add independently reviewed domain text,
annotation agreement/adjudication, more languages, class balance and source-level
split groups. This tiny synthetic fixture establishes none of those properties.

## Run

From the parent ontology directory:

```bash
python semantic_evaluation.py
python semantic_evaluation.py --output /tmp/room-semantic-baseline.json
python semantic_evaluation.py --predictions /path/to/predictions.jsonl --output /tmp/model-report.json
```

Default input paths resolve relative to the script. `--benchmark` accepts another
annotated fixture; `--split` selects train, validation or test after validating
all splits for leakage. The baseline reuses the current RoomClassifier without
changing taxonomy, vocabulary or policy handling. Candidate assignments count as
positive predictions even when `requires_context_review` is true; review flags
are retained and reported separately, not treated as semantic disambiguation.
The command exits successfully when evaluation completes, even if scores are low.

External predictions are a JSON array or JSONL with exactly one record for every
selected case, no missing/extra IDs:

```json
{"case_id":"private","taxonomy_version":"1.3.0","concept_ids":["private_show"],"requires_context_review":true}
```

The review flag is optional. Reports expose its coverage so an omitted flag does
not imply confident classification. A semantic model can produce these records
outside this evaluator; no model is bundled and no inference endpoint is called.
Unknown IDs, L0 labels, duplicate records/labels and mismatched taxonomy versions
are rejected. Preserve raw text for modeling; NFKC/casefold normalization is used
only for leakage checks.

## Reading scores

- **Concept micro precision/recall/F1:** direct IDs; wrong siblings and extra labels count.
- **Category scores:** derive L1 codes from concept IDs; assess broader branch membership.
- **Hierarchy-aware scores:** expand direct concepts to their ancestors, exclude L0, and give partial credit for a correct broad parent.
- **Exact match:** the predicted label set must equal the complete gold set.
- **Macro F1:** average labels with either gold or predicted support; per-label support is shown.
- **Taxonomy-negative false-positive rate:** fraction of cases with empty gold that receive any candidate assignment; not an unsafe-content or policy FPR. Reported as null when no such cases exist.
- **Slices and per-case errors:** reveal where aggregate metrics hide missed paraphrases or benign keyword senses. Review flags do not remove false positives from scoring.

All-empty sets have zero label precision/recall/F1, while exact match is one.
No slice/metric threshold is tuned here. Small slices and differing label sets
make these numbers unsuitable for production-quality or external-dataset claims.

## Recorded keyword baseline

[baseline_summary.json](baseline_summary.json) records the fixture and vocabulary
SHA-256 hashes so results are tied to the exact inputs. On these 28 cases,
concept micro-F1 is **0.679**, concept exact match is **15/28**, and **8/10**
taxonomy-negative cases receive candidate labels. Seventeen cases carry a review
flag. These are deliberate context challenges, not representative deployment
rates. This historical keyword summary predates the [local-model comparison](LOCAL_BASELINE.md);
perfect-prediction unit tests validate metric arithmetic only. Re-run the command after changing the fixture or
vocabulary rather than treating the historical summary as current results.

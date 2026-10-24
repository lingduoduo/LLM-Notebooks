# Adult Content Ontology v1

A minimal, extensible multi-label ontology and keyword matcher for room subjects and up to five tags.

## Run

```bash
python build.py --init
python build.py --subject 'Private show with cosplay' --tags bdsm lingerie
python build.py --import-words /path/to/LDNOOBW/en --language en
```

Importing a dictionary creates `candidate_terms.csv` with *unmapped* terms. Review and assign sexual-content concept IDs before adding terms to `keywords_v1.csv`. This prevents profanity, identity words, and benign ambiguous terms from being automatically treated as adult content.

## Files

- `taxonomy_v1.json`: concept nodes and IS_A edges.
- `taxonomy_schema.json`: JSON Schema for taxonomy file.
- `keywords_v1.csv`: small curated *illustrative* starter vocabulary; not a comprehensive NSFW list.
- `sample_output.json`: sample classifier output.
- `build.py`: generator, simple matcher, and candidate importer.

## Limitations

This is a deterministic baseline, not a trained classifier. Keyword matches are evidence, not calibrated probabilities. It supports Unicode NFKC/casefold but does not handle morphology, misspellings, obfuscation, multi-language segmentation, negation, or semantic disambiguation. For production, add language-specific tokenization, review, provenance, evaluation, and a separate safety/policy layer.

## Classify a batch

Load files and prepare a matcher once, then reuse it across records:

```python
from build import Matcher, load

parents, rows = load()
matcher = Matcher(parents, rows)
result = matcher.match('Private show with cosplay', ['bdsm', 'lingerie'])
```

Preparation normalizes keywords, compiles their boundary patterns, and resolves
ancestor sets. Output order, Unicode normalization, overlapping keyword matches,
and the five-tag limit are preserved. The existing
`match(subject, tags, parents, rows)` function remains available for single calls;
it prepares a new matcher each time. Construct a new matcher after editing the
ontology or vocabulary.

Invalid parent references, cycles, duplicate node IDs when loading, empty keywords,
unknown keyword concepts, and invalid `context_required` flags raise `ValueError`.
This validation does not replace full JSON Schema or ontology-edge validation.

A local comparison over 1,000 generated records using the supplied 22 keywords
produced identical results to the original implementation. Best of five runs:
original 71 ms, reused matcher 28 ms (approximately 2.5× faster), compatibility
wrapper 68 ms. These measurements exclude file loading and matcher preparation
for the reusable path; throughput depends on text length and vocabulary size.
Matching still visits every keyword for each field, but runs the boundary regex
only when the keyword occurs as a substring. Over 5,000 randomized records this
gave identical output and cut a reused-matcher batch of 1,000 from 38 ms to
12 ms (approximately 3.2× faster). Cached ancestor sets use
memory proportional to the sum of node depths, which suits this small taxonomy.

## Verify

From this directory:

```bash
python3 -W error::ResourceWarning -m unittest discover -v
```

Tests cover the supplied sample output, Unicode and word boundaries, the tag
limit, invalid data, generated artifact compatibility, and candidate import
normalization/deduplication. The package uses only Python's standard library.

## Candidate import memory

Dictionary files are read one line at a time. Deduplication retains one key per
unique `(keyword, language)` pair; the full input file is no longer retained.
A local 400,000-line input containing repeated terms produced byte-identical CSV
output with peak Python-traced memory reduced from 34.42 MiB to 0.42 MiB. Unique
vocabularies still require memory proportional to their unique term count.
Unicode line separators remain supported. Importing the destination
`candidate_terms.csv` itself (including filesystem aliases) raises `ValueError`.
Imports append incrementally; an interrupted import may retain completed rows,
and rerunning it deduplicates those rows.

## Related Spark SQL

`../social_follow_graph_spark.sql` computes shared-follower similarities and
recommendations. Its pair join generates `d * (d - 1) / 2` pairs for a user with
`d` follows. Temporary views are logical definitions, so the script caches the
two views read by several later steps: deduplicated `follow_edges` (one source
scan and shuffle instead of one per consumer) and `broadcaster_similarity` (the
quadratic pair join is computed once). On local Spark 4.1.1 with a synthetic
6,000-edge graph, every view matched the uncached script exactly and the script
re-ran cleanly in the same session. No runtime claims are made without a
representative dataset. Budget cache storage (it spills to disk) and `UNCACHE`
when the session finishes. Degree caps and approximate pair generation change
results and require an explicit quality/performance tradeoff.

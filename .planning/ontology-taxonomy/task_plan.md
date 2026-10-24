# Hierarchical room taxonomy
1. Inspect current source and compatibility surfaces: complete.
2. Add tests for L0+L1–L3 metadata, stable IDs, multi-branch room assignment, evidence and distinct-room counts: complete.
3. Implement taxonomy metadata/validation, room API, schema and generated assets: complete.
4. Document API/version changes, run full checks and independent review: complete.

Scope: preserve existing load()/Matcher/match() output contract and concept IDs. New category_id codes BODY/ACT/KINK/FORMAT/PRODUCT/SOCIAL/IDENTITY/COMMERCE, optional L0 root, three classification levels. Keep existing taxonomy_v1.json path for compatibility; version metadata identifies new release. New room API retains raw evidence, supports explicit self-declared identity, and derives distinct room category counts.

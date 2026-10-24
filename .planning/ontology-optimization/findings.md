- Entire ontology-use-case directory is initially untracked; preserve existing artifacts.
- Python repeats normalization/regex construction per field and call; ancestor traversal can loop forever on cycles.
- load() leaves CSV file lifetime implicit; text files use platform-default encoding in places.
- SQL pair enumeration is quadratic per user; temporary views do not guarantee materialization. Changing degree filters would change recommendation semantics, so retain SQL pending workload evidence.

Follow-up: executable files appear unchanged; PDF/ZIP/supporting duplicate CSV removed by user. Candidate import retained entire dictionary and self-import appended CSV lines as bogus terms. Streaming reduces repeated-input memory; samefile guard protects direct paths and filesystem aliases.

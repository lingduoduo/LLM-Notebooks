# Findings

Baseline: prior runs report 19 passing tests and one historical source-hash mismatch. Rechecking current state.

Baseline checks: 17 Python files parse; 17 JSON files parse; six CLI help commands and offline role/scenario listing pass. Pytest: 19 passed, 1 failed (historical source hash). Ruff: two unused-symbol errors in evaluation.py.

Confirmed review findings: role tool authorization is not enforced; evaluator can treat failed required tool as successful when another tool succeeds; counted draft can differ from final delivery; judge does not guarantee swapped order; resume ignores model/task compatibility. Investigating precise reproductions and severity.

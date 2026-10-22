# Local model validation implementation plan

**Goal:** Harden the existing local inference API, test the client without weights, and repair documentation, as approved in chat.

**Architecture:** Validate public inputs in client.py before calling the unchanged upstream inference modules. HTTP parsing errors remain 400; inference failures remain 500.

**Tech stack:** Python, unittest/pytest, stdlib HTTP server.

- [x] Add client regression tests for malformed questions, model names, supported question shapes, and response cleanup; run and observe failures.
- [x] Validate choice, score, and noul definitions while preserving structured instructions and optional noul criteria.
- [x] Add HTTP regressions for invalid UTF-8 and invalid model fields; observe failures, then implement parsing checks.
- [x] Repair the missing RLCD reference and document accepted question forms and error behavior.
- [x] Run the complete Jev_LocalModel test suite and inspect edits. No model weights are installed, so real inference remains unverified.

Work stays in the current workspace because the entire existing project is untracked and must be preserved. No commits or dependency installations are needed.

Validation: `python -m pytest Jev_LocalModel/tests -q` — 19 passed, 25 subtests passed. Python compilation and README local-link checks passed. No checkpoint weights are present; live inference was not run. Invalid UTF-8 already returned 400 through ValueError handling, so no parsing change was necessary for that case.

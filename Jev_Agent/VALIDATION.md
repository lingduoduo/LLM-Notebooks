# Validation — 2026-10-19

- Python: 11 tests passed with `python -m unittest discover -s Jev_Agent/tests -v`.
- Node: 15 tests passed with `node --test Jev_Agent/tests/extension.test.mjs`.
- Both notebooks passed nbformat validation and Python cell syntax checks (excluding the pip magic).
- Pi notebook setup resolved the extension from both the repository root and Jev_Agent directory.
- Installed Pi CLI 1.3.0 into `/tmp/jev-agent-pi-validation` using npm with `--ignore-scripts`; no repository dependency files were changed.
- Real Pi RPC returned accessibility skill selection, allow for the git-status example, and block for the destructive-command example. Gate commands classify arguments without executing those shell commands.
- Executed every Pi notebook code cell through a Python Jupyter kernel from Jev_Agent. Replaced only the in-memory pip-install cell with imports of the already installed nbformat and nbclient packages. Source notebook outputs were not overwritten.

Pi integration used the extension's explicit `offline-simulation` because TYPESAFE_API_KEY was unset. This verifies Pi loading, RPC transport and the offline notebook flow; it does not verify TypeSafe API behavior or model accuracy. The runtime was Node.js 23.7.0, which emitted an experimental type-stripping warning during tests; the notebook recommends Node.js 24+.

DSH end-to-end execution remains unverified: the external dsh-jev-decision project and original validation archive were not found, and DSH_JEV_APP was unset. Set DSH_JEV_APP to that external project to continue. Live TypeSafe checks also require TYPESAFE_API_KEY in the execution environment.

- Added a regression for a Pi descendant that ignores SIGTERM and retains an output pipe; process-group cleanup now escalates even after the Pi leader exits. POSIX descendant cleanup is tested; Windows descendant cleanup is not covered.

---
name: calisim-check
description: Select and run checks for a calisim change, diagnose test or environment failures, and report validation evidence.
---

# Validate a change

Inspect the task, `git status --short`, the diff, and relevant tests. Read
[the development guide](../../../docs/development/README.md) for the current
commands and optional-dependency caveats.

Choose checks by the behavior affected:

| Change | Useful starting check |
| --- | --- |
| Agent configuration or hooks | `python3 scripts/agents/check_setup.py` and standard-library tests under `scripts/agents/tests/` |
| Utilities, metrics, demonstration models | The corresponding `tests/test_*.py` module |
| A scientific backend | A selected case in `tests/functional/test_<task>.py` |
| Shared workflow or data contracts | Relevant callers across affected task families |
| Hydra or TorchX | Corresponding configuration or orchestration tests with the extra installed |
| Sphinx sources | Sphinx HTML build plus inspection of warnings |
| Contributor Markdown | Local links, Markdown structure, and command accuracy |

Run targeted checks first, then broaden when the affected behavior or a failure
justifies it. Include Ruff lint and format checks for changed Python files.
Agent tooling can be checked without importing the scientific test fixtures.

Distinguish missing dependencies and collection failures from test assertion
failures. The `--torch` flag enables marked cases; it does not install extras or
prevent every collection-time import. Avoid changing scientific behavior,
dependency constraints, or tolerances simply to get a passing run.

Report exact commands and outcomes, skipped checks, artifacts produced, and any
remaining limits. A passing subset does not establish that the whole suite
passes. Make changes only when the user's task authorizes them.

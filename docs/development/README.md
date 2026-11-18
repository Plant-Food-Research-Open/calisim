# AI-assisted development

Start an assistant task with [AGENTS.md](../../AGENTS.md). It contains the shared
repository instructions; [CLAUDE.md](../../CLAUDE.md) and
[Copilot instructions](../../.github/copilot-instructions.md) point to it.
The [AI policy](../../AI_POLICY.md) covers human responsibility, disclosure, and
scientific verification.

Use the [architecture map](architecture.md) to find an implementation and its
tests. Copy the [task template](task-template.md) into an issue, task brief, or
working note when a change benefits from explicit acceptance criteria. Small
fixes can state those criteria directly in the task.

See [agent setup](agent-setup.md) for reusable skills, specialist subagents,
scoped rules, lifecycle hooks, and checks for the agent configuration itself.

## Set up the environment

Run commands from the repository root. Use Python 3.11 or 3.12 and an installed
Poetry CLI. Reuse a working environment when one is already configured.

```bash
poetry env use 3.11
poetry install --with dev,docs --without experimental -E hydra -E torchx
```

Hydra and TorchX are included here because their test modules import them during
collection. Experimental ELFI is a separate Git dependency and is excluded from
this setup. Install that group when working on `calisim/experimental/`.

For Torch-based engines, install the additional extra:

```bash
poetry install --with dev,docs --without experimental -E hydra -E torchx -E torch
```

Use `poetry install` to follow the lockfile. Use `poetry update` only when changing
dependencies is part of the task. If installation or imports fail, record the
actual error and resolve the environment before interpreting test results.

## Choose checks for the change

Start with the smallest test that exercises the changed behavior. Add a
regression test for a defect or new executable behavior when it provides useful
coverage. Documentation-only edits do not require new Python tests.

```bash
# Baseline checks without running calibration workflows
poetry run ruff check calisim tests examples
poetry run ruff format --check calisim tests examples
poetry run pytest tests/test_utils.py tests/test_discrepancies.py tests/test_example_models.py

# Example: exercise the Optuna cases after an optimisation backend change
poetry run pytest tests/functional/test_optimisation.py -k optuna

# Broader checks excluding functional workflows
poetry run pytest --ignore=tests/functional

# Full suite, including Torch-marked cases; requires the torch extra
poetry run pytest --torch
```

Functional tests run real scientific backends and may produce plots and other
artifacts. They can take substantially longer than the baseline. The custom
`--torch` flag enables `torch`-marked tests; it does not install dependencies or
prevent all imports during collection. Without that flag, those marked tests
are skipped. Broad suite commands above assume Hydra and TorchX are installed.

CI tests Python 3.11 and 3.12. Its push job excludes `tests/functional`; other
test events run the full suite with `--torch`. See
[test.yaml](../../.github/workflows/test.yaml) for the current matrix, extras,
parallelism, and report options.

Run the repository's existing pre-commit hooks on the changed files, for example:

```bash
poetry run pre-commit run --files AGENTS.md CLAUDE.md .github/copilot-instructions.md
```

Replace that file list with the files in your patch. Hooks include formatting,
type checks, spelling, security, and whitespace checks as applicable; some hooks
modify files. Inspect those changes and rerun affected checks. The first hook
run may need network access to prepare its environments.

For changes to the Sphinx documentation:

```bash
poetry run sphinx-build -b html docs/source docs/build/html
```

Check warnings as well as the exit status. Files in `docs/development/` are
repository contributor documentation and are linked from the README; the
published Sphinx sources are in `docs/source/`. Several community documents have
copies there, so keep corresponding content consistent when editing them.

## Complete and hand off a task

- Confirm each acceptance criterion against code, documentation, or executed
  checks. State any assumption that still needs review.
- Inspect `git diff --check`, `git diff`, and `git status --short`, including new
  files. Keep generated experiment outputs and unrelated edits out of the patch.
- Summarize the behavior changed and why, with exact verification commands and
  results. Identify checks that were skipped or blocked and what that limits.
- Disclose material AI assistance in submitted work using the
  [pull request template](../../.github/pull_request_template.md). State whether
  human review is pending or complete.

Update the shared instructions when the architecture or commands change. Keep
tool-specific entry files short so they continue to point to the same guidance.

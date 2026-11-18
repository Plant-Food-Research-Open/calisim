# Agent instructions for calisim

calisim is a Python library for calibrating and evaluating simulation models.
It provides a common workflow over third-party scientific libraries. Preserve
that shared interface when changing an individual backend.

## Start here

- Read [README.md](README.md) for the public workflow and
  [AI_POLICY.md](AI_POLICY.md) for responsibility and disclosure requirements.
- Use the [architecture map](docs/development/architecture.md) to find the
  relevant implementation, tests, examples, and API documentation.
- See the [development guide](docs/development/README.md) for environment setup
  and checks. Python 3.11 and 3.12 are supported; dependencies use Poetry and
  `poetry.lock`.
- For substantial or ambiguous work, use the
  [task template](docs/development/task-template.md) to record acceptance
  criteria, scope, and a verification plan.

## Skills, specialists, and scoped rules

Read [agent setup](docs/development/agent-setup.md) for discovery, hooks, and
maintenance. Reusable skills live in `.agents/skills/`:

- `calisim-backend` for backend, specification, and workflow changes.
- `calisim-check` for selecting checks and reporting validation evidence.
- `calisim-scientific-review` for read-only numerical and scientific review.
- `calisim-docs` for examples, tutorials, and contributor documentation.

Follow subtree `AGENTS.md` rules in `calisim/`, `tests/`, `examples/`, and `docs/`
when working there. Specialist roles are defined in
[agent-roles.md](docs/development/agent-roles.md): `code-explorer`,
`backend-implementer`, and `scientific-reviewer`. Use them when a bounded task
benefits from delegation. Assign disjoint file ownership for parallel edits and
keep responsibility for integration in the parent agent. Small changes do not
need delegation.

## Working approach

1. Inspect `git status --short`, the affected code, and nearby tests before
   editing. Preserve existing contributor changes.
2. Establish the expected behavior and reproduce a reported defect when
   practical. State assumptions that affect the result; ask when a missing
   requirement prevents a sound implementation.
3. Make a focused change using existing abstractions. Update affected public
   examples and documentation when behavior changes.
4. Run checks appropriate to the change. Report exact commands, outcomes,
   skipped checks, and environment limitations. Distinguish pre-existing
   failures from regressions when evidence supports that distinction.
5. Review the diff for unrelated edits and generated artifacts. Hand off the
   result with its validation evidence and any remaining human review.

Carry authorized tasks through implementation and verification. Keep code
review requests read-only unless edits are requested. Follow the contributor's
instructions and repository permissions for publishing and release work.

## Commit messages

Use Conventional Commits when committing changes: `type(scope): summary`, with
an optional scope and a short imperative summary. For example,
`chore(agents): add repository agent tooling`. Choose a type that describes the
change, such as `feat`, `fix`, `docs`, `test`, `ci`, or `chore`. Mark breaking
changes with `!` or a `BREAKING CHANGE:` footer. The existing
`conventional-pre-commit` hook checks commit messages at the `commit-msg` stage.

## Architecture and compatibility

- Public task classes and their Pydantic specifications live in
  `calisim/<task>/implementation.py`, with exports in `__init__.py`.
- Backend implementations live in `*_wrapper.py` and generally extend
  `CalibrationWorkflowBase` or a specialized base in `calisim/base/`.
- Preserve the public `specify().execute().analyze()` chain, hooks, artifact
  tracking, and parameter-estimate access through `CalibrationMethodBase`.
- Keep backend-specific behavior in its wrapper. Shared behavior belongs in an
  existing base class or utility when its contract applies across backends.
- Preserve lazy engine resolution through `BASE_IMPLEMENTATIONS` and
  `get_implementations()`, including external entry points under
  `calisim.external.<task>`.
- Keep optional backends optional. Avoid introducing unconditional imports of
  Torch, Hydra, TorchX, or experimental ELFI into common import paths.
- Treat public exports, engine names, specification fields, defaults, and
  objective signatures as compatibility-sensitive. Explain intended changes.

## Python and scientific conventions

- Follow `pyproject.toml`, `.pre-commit-config.yaml`, and nearby code. Ruff uses
  tabs for indentation, double quotes, and an 88-character line length. Add
  type annotations and use the existing `Args`/`Returns` docstring style.
- Use the Pydantic models in `calisim/data_model/` for shared specifications;
  put task-specific fields in the task's specification subclass.
- Check parameter ordering, parameter types and bounds, array shapes, output
  labels, and objective direction at backend boundaries. Preserve distinctions
  between simulator outputs, discrepancies, and log-likelihoods.
- Use explicit random seeds where supported in reproducibility-sensitive
  examples and tests. Check backend-specific seed settings as well as the
  workflow's NumPy generator; one seed does not necessarily seed every backend.
- Use small fixtures and justified numerical tolerances for stochastic tests.
  Do not loosen tolerances or change scientific defaults just to make a check
  pass. Explain substantive analytical or algorithmic changes.
- Verify third-party APIs against the installed or locked dependency versions.
  Avoid dependency upgrades and lockfile changes unrelated to the task.

## Verification

Run from the repository root in a configured Poetry environment:

```bash
poetry run ruff check calisim tests examples
poetry run ruff format --check calisim tests examples
poetry run pytest tests/test_utils.py tests/test_discrepancies.py tests/test_example_models.py
```

Select relevant tests beyond this baseline. Backend changes usually require
the corresponding `tests/functional/test_<task>.py` cases; shared-base changes
need coverage of the affected callers. Use `--torch` only with the `torch`
extra installed. Hydra and TorchX test modules import optional packages at
collection time; the marker flag does not make missing dependencies available.
See the development guide for the full environment and suite commands.

For documentation-only changes, check links, formatting, and command accuracy;
run Sphinx when changing its sources. Run applicable pre-commit hooks on changed
files before handing off code or configuration changes. Report checks that
could not run rather than claiming success.

For agent tooling changes, run these checks without loading scientific fixtures:

```bash
python3 scripts/agents/check_setup.py
python3 -m unittest discover -s scripts/agents/tests -v
poetry run ruff check scripts/agents
poetry run ruff format --check scripts/agents
```

## Repository hygiene

- Keep simulation outputs, plots, databases, logs, coverage reports, caches,
  virtual environments, and built distributions out of patches unless they are
  intentional deliverables. Tests commonly write to `tests/outdir/`; examples
  commonly write to `examples/outdir/`.
- Preserve source observations and research provenance. Do not invent results,
  references, API behavior, or test evidence.
- Keep credentials and restricted research data out of prompts and commits;
  follow the AI policy for data-owner authorization.
- Leave version bumps, tags, and release artifacts to the release task. A
  documentation or tooling change alone does not require a package release.

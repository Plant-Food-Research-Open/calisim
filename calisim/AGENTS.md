# Library rules

Use the root instructions together with these rules for `calisim/`.

- A task's `implementation.py` owns its public facade, Pydantic specification,
  and lazy engine mapping. Put backend-specific changes in `*_wrapper.py`.
- Preserve `CalibrationMethodBase` delegation, workflow and objective hooks,
  artifact tracking, parameter-estimate access, and public package exports.
- When changing shared bases, inspect callers across task families. Parameter
  conversion must preserve ordering, categorical values, constants, bounds,
  batching, and simulation identifiers.
- Keep optional libraries out of common import paths. Check behavior both with
  the changed optional backend and without importing it.
- Confirm whether a backend consumes simulator output, a discrepancy to
  minimize, or a log-likelihood to maximize. Check shapes and labels explicitly.
- See [the backend skill](../.agents/skills/calisim-backend/SKILL.md) for an
  integration workflow and [the architecture map](../docs/development/architecture.md)
  for relevant files.

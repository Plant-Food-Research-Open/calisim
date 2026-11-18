---
name: calisim-docs
description: Update calisim examples, tutorials, API references, or contributor documentation and verify the relevant commands and outputs.
---

# Documentation and examples

Read [the documentation rules](../../../docs/AGENTS.md) and
[the example rules](../../../examples/AGENTS.md) when applicable. Match the
documentation to the source code and locked dependency versions.

Published documentation lives under `docs/source/`; contributor guidance lives
under `docs/development/`. Sphinx API pages are grouped by task. Community
contribution documents may have corresponding root copies.

For changed public behavior, align the relevant wrapper or specification,
example script, tutorial, and API reference. State required extras, objective
meaning, shapes, and reproducibility settings where they help the reader use
the example correctly.

Run a changed example when its runtime and dependencies permit. A rendered page
does not verify a calibration result. Describe proposed or unexecuted examples
accurately rather than presenting them as tested results.

For Sphinx source changes, run
`poetry run sphinx-build -b html docs/source docs/build/html` and inspect warnings.
For agent contributor documentation, run `python3 scripts/agents/check_setup.py`
to check local links. Keep generated builds and experiment outputs out of the
patch and report the checks actually completed.

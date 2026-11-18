# Test rules

Use the root instructions together with these rules for `tests/`.

- Reuse `tests/conftest.py` fixtures for demonstration models, parameter
  specifications, metrics, output directories, and calibration helpers.
- `pytest.ini` enables strict markers and strict xfails. Register a new marker
  before using it; an unexpected passing xfail is a failure.
- Torch-marked tests require `--torch`; the flag does not install packages.
  Hydra and TorchX test modules import their extras during collection.
- Backend integration coverage belongs in `tests/functional/test_<task>.py`.
  Prefer a small regression case before a complete calibration run.
- Use explicit backend seed settings where supported. Justify numerical
  tolerances and test scientific invariants instead of exact stochastic output.
- Use temporary paths for new tests that create artifacts. Existing functional
  fixtures write to `tests/outdir/`; keep those results out of patches.
- Agent tooling tests live in `scripts/agents/tests/` and run with standard
  library unittest, avoiding scientific fixture imports.

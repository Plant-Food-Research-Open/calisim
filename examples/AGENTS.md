# Example rules

Use the root instructions together with these rules for `examples/`.

- Keep examples small, runnable from the repository root, and consistent with
  the corresponding public task API and locked dependency versions.
- State required optional extras. Specify seeds and modest iteration counts
  where supported so demonstrations are reproducible and practical.
- Preserve the scientific meaning of observations, parameters, discrepancy
  metrics, and objective direction. Identify synthetic data as synthetic.
- Use `get_examples_outdir()` for normal example artifacts. Do not commit the
  resulting plots, databases, or calibration outputs.
- When changing public behavior, update the corresponding example and tutorial
  together. Run the example when its runtime and dependencies permit; describe
  validation limits if it was only inspected.

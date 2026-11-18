# Architecture map

calisim presents a common interface to calibration, inference, and evaluation
libraries. Follow an existing task and backend when extending the library;
[optimisation](../../calisim/optimisation/implementation.py) and its
[Optuna wrapper](../../calisim/optimisation/optuna_wrapper.py) provide a concrete
example.

## Workflow and extension points

1. A user constructs a Pydantic task specification, such as
   `OptimisationMethodModel`, and supplies a calibration function.
2. A public task class, such as `OptimisationMethod`, extends
   `CalibrationMethodBase`. The task's `get_implementations()` combines built-in
   engine mappings with installed external implementations.
3. `CalibrationMethodBase` resolves the selected `module:Class` lazily, or uses
   an explicitly supplied implementation class or instance.
4. The public `specify().execute().analyze()` methods delegate to the backend
   and return the public object for chaining. Their pre/post hooks surround
   each stage.
5. Backends extend `CalibrationWorkflowBase` or specialized bases. They prepare
   their engine, invoke the calibration function, and expose artifacts and
   parameter estimates through the shared interface.

The shared calibration call takes `parameters`, `simulation_id`,
`observed_data`, and configured keyword arguments. Depending on the task, it
returns simulator outputs, a discrepancy, or a likelihood value. Batched
execution changes the shapes of inputs and outputs; inspect the relevant wrapper
and fixtures before assuming a shape. Calibration-function hooks are separate
from workflow-stage hooks.

External engines register entry points under `calisim.external.<task>`.
Built-in mappings use `module:Class` strings, allowing an engine to be selected
without importing every backend. Preserve that boundary when adding engines.

## Where changes belong

| Area | Location and purpose |
| --- | --- |
| Shared workflow | [calisim/base/](../../calisim/base/): dispatch, hooks, sampling helpers, and specialized workflow bases |
| Shared data contracts | [calisim/data_model/](../../calisim/data_model/): Pydantic specifications, parameter types, distributions, and estimates |
| Task facade and specification | `calisim/<task>/implementation.py`, with public exports in its `__init__.py` |
| Scientific backend | `calisim/<task>/*_wrapper.py`: adapters to individual third-party libraries |
| Discrepancies and designs | [calisim/statistics/](../../calisim/statistics/): metrics, likelihoods, and experimental designs |
| Emulator adapters | [calisim/estimators/](../../calisim/estimators/): estimator interfaces for surrogate libraries |
| Configuration | [calisim/config/](../../calisim/config/): optional Hydra integration |
| Job execution | [calisim/orchestration/](../../calisim/orchestration/): optional TorchX integration |
| Demonstration simulators | [calisim/example_models/](../../calisim/example_models/): SIR, Lotka–Volterra, Lorenz, and oscillator models |
| Shared utilities | [calisim/utils/](../../calisim/utils/): calibration-function adaptation, identifiers, paths, and other helpers |
| Experimental integrations | [calisim/experimental/](../../calisim/experimental/): integrations such as ELFI |
| Focused tests | [tests/](../../tests/): utility, metric, model, estimator, configuration, and orchestration checks |
| Backend workflows | [tests/functional/](../../tests/functional/): end-to-end checks grouped by task |
| Shared test setup | [tests/conftest.py](../../tests/conftest.py): model and specification fixtures, calibration helpers, and Torch selection |
| User examples | [examples/](../../examples/): scripts grouped by task and backend |
| Published documentation | [docs/source/](../source/): Sphinx tutorials, API references, and community pages |
| Automation and dependencies | [.github/workflows/](../../.github/workflows/), [pyproject.toml](../../pyproject.toml), and `poetry.lock` |

Task families include optimisation, Bayesian calibration, ABC, simulation-based
inference, history matching, evolutionary methods, surrogate modelling, active
learning, experimental design, sensitivity, uncertainty, reliability, and
quadrature. Consult each task's current engine mapping for supported backends.

## Adding a backend

- Inspect an existing wrapper for the same task and verify the dependency's API
  against the locked version.
- Implement the relevant workflow stages and use existing shared helpers for
  parameter conversion, objective calls, artifacts, and estimates.
- Add the lazy mapping to the task's `BASE_IMPLEMENTATIONS`. Add specification
  fields only when the task needs a new public configuration option.
- Declare a dependency or optional extra when needed. Update the lockfile
  through Poetry and include an appropriate third-party license in
  [third_party/](../../third_party/).
- Add meaningful backend tests and a small example. Update task exports when
  adding public symbols and the corresponding Sphinx API reference when needed.
- Check that common imports still work without the new optional backend.

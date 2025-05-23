Changelog
=========

[0.6.0] - 2025/04/27
--------------------

Added
^^^^^

* Added unit and functional tests for calisim modules.

[0.5.0] - 2025/04/25
--------------------

Added
^^^^^

* Implemented likelihood-based discrepancy metrics.
* Included log likelihood for fully Bayesian calibration examples.

Fixed
^^^^^

* Refactored logic for best parameter estimates to always set estimates.
* Debugged parallelisation and plot rendering for pyABC wrapper.

[0.4.0] - 2025/04/06
--------------------

Added
^^^^^

* Implemented Lorenz 95 and anharmonic oscillator example models.
* Added replicate sampling functionality, particularly for stochastic simulators.
* Incorporated several new discrepancy functions.
* Included experimental support for the Ensemble Kalman Filter.

[0.3.1] - 2025/03/30
--------------------

Added
^^^^^

* Implemented experimental support for the ELFI library.
* Implemented ELFI wrapper for ABC.
* Implemented ELFI wrapper for likelihood-free inference.

Fixed
^^^^^

* Added multi-objective support for Optuna.

[0.3.0] - 2025/03/24
--------------------

Fixed
^^^^^

* Removing Python 3.13 support.

Fixed
^^^^^

* Unpinning PyMC for Python 3.13 support.

[0.2.3] - 2025/03/24
--------------------

Fixed
^^^^^

* Unpinning PyMC for Python 3.13 support.

[0.2.2] - 2025/03/24
--------------------

Fixed
^^^^^

* Downgrading pygpc for Python 3.13 support.

[0.2.1] - 2025/03/21
--------------------

Fixed
^^^^^

* Widened range of supported versions of Python.

[0.2.0] - 2025/03/20
--------------------

Added
^^^^^

* Implemented EvoTorch wrapper.
* Included calisim logo.
* Bundled in TorchX as optional dependency.
* Incorporated Docker image build.

Changed
^^^^^^^

* Fleshed out documentation in README.

Fixed
^^^^^

* Added checks for the optional importing of PyTorch.
* Included missing API documentation for models.

[0.1.0] - 2025/03/17
--------------------

Added
^^^^^

* Open sourcing calisim Python package.
* PyPI deployment.
* ReadTheDocs deployment.

Changed
^^^^^^^

* Configuring CI/CD builds for testing, linting, and publishing.

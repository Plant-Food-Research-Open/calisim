Changelog
=========

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

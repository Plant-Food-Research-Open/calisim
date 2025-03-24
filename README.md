<div align="left"><img src="https://raw.githubusercontent.com/Plant-Food-Research-Open/calisim/main/docs/assets/calisim_logo.png" width="400" height="130"/></div>

______________________________________________________________________

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/calisim)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Lint](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/lint.yaml)
[![Test](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/test.yaml)
[![Publish](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/publish.yaml/badge.svg?branch=main)](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/publish.yaml)
[![Build](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/Plant-Food-Research-Open/calisim/actions/workflows/build.yaml)
[![Run with Docker](https://img.shields.io/badge/run%20with-docker-0db7ed?labelColor=000000&logo=docker)](https://www.docker.com/)

[**PyPI**](https://pypi.python.org/pypi/calisim)
| [**Documentation**](https://calisim.readthedocs.io)
| [**API**](https://calisim.readthedocs.io/en/latest/api_reference/index.html)
| [**Changelog**](https://calisim.readthedocs.io/en/latest/changelogs/changelog.html)
| [**Examples**](https://github.com/Plant-Food-Research-Open/calisim/tree/main/examples)
| [**Releases**](https://github.com/Plant-Food-Research-Open/calisim/releases)
| [**Docker**](https://github.com/Plant-Food-Research-Open/calisim/pkgs/container/calisim)

*A toolbox for the calibration and evaluation of simulation models.*

# Table of contents

- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Features and Functionality](#features-and-functionality)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage with Docker](#usage-with-docker)
- [Communication](#communication)
- [Contributions and Support](#contributions-and-support)
- [License](#license)

# Introduction

calisim is an open-source, low-code model calibration library that streamlines and standardises your workflows, while aiming to be as flexible and extensible as needed to support more complex use-cases. Using calisim will speed up your experiment cycle substantially and make you more productive.

calisim is primarily a wrapper around popular libraries and frameworks including Optuna, PyMC, scikit-learn, and emcee among many others. The design and simplicity of calisim was inspired by the scikit-learn and PyCaret libraries.

# Features and Functionality

* A standardised and streamlined interface to multiple calibration procedures and libraries.
* A low-code library, allowing modellers to rapidly construct multiple workflows for many calibration procedures.
* An object-oriented programming architecture, allowing users to easily extend and modify calibration workflows for their own complex modelling use-cases.
* An unopinionated approach to working with simulation models, allowing users to calibrate both Python-based and non-Python-based models.
* Optional integration with PyTorch for access to more sophisticated Gaussian Process and deep learning surrogate models, state-of-the-art evolutionary algorithms, and deep generative modelling for simulation-based inference.

# Quickstart

```
# Load imports
import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.optimisation import OptimisationMethod, OptimisationMethodModel
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

# Get model
model = LotkaVolterraModel()
observed_data = model.get_observed_data()

# Specify model parameter distributions
parameter_spec = ParameterSpecification(
	parameters=[
		DistributionModel(
			name="alpha",
			distribution_name="uniform",
			distribution_args=[0.45, 0.55],
			data_type=ParameterDataType.CONTINUOUS,
		)
	]
)

# Define objective function
def objective(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(
		alpha=parameters["alpha"],
		beta=0.024, h0=34.0, l0=5.9,
		t=t, gamma=0.84, delta=0.026,
	)

	simulated_data = model.simulate(simulation_parameters).lynx.values
	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy

# Specify calibration parameter values
specification = OptimisationMethodModel(
	experiment_name="optuna_optimisation",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=get_examples_outdir(),
	method="tpes",
	directions=["minimize"],
	n_iterations=100,
	method_kwargs=dict(n_startup_trials=50),
	calibration_func_kwargs=dict(t=observed_data.year),
)

# Choose calibration engine
calibrator = OptimisationMethod(
	calibration_func=objective, specification=specification, engine="optuna"
)

# Run the workflow
calibrator.specify().execute().analyze()

# View the results
result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")
print(f"Parameter estimates: {calibrator.get_parameter_estimates()}")
```

# Installation

The easiest way to install calisim is by using pip:

```
pip install calisim
```

calisim's default installation will not include all optional dependencies. You may be interested in one or more extras:

```
# Install PyTorch extras
pip install calisim[torch]

# Install Hydra extras
pip install calisim[hydra]

# Install TorchX extras
pip install calisim[torchx]

# Install multiple extras
pip install calisim[torch,hydra,torchx]
```

# Usage with Docker

You may also want to execute calisim inside of a Docker container. You can do so by running the following:

```
# Change the image version as needed
export CALISIM_VERSION=latest

# Get docker-compose.yaml file
wget https://raw.githubusercontent.com/Plant-Food-Research-Open/calisim/refs/heads/main/docker-compose.yaml

# Pull the image
docker compose pull calisim

# Run an example
docker compose run --rm calisim python examples/optimisation/optuna_example.py
```

# Communication

Please refer to the following links:

- [GitHub Discussions] for questions.
- [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/Plant-Food-Research-Open/calisim/discussions
[GitHub issues]: https://github.com/Plant-Food-Research-Open/calisim/issues

# Contributions and Support

Contributions are more than welcome. For general guidelines on how to contribute to this project, take a look at [CONTRIBUTING.md](https://github.com/Plant-Food-Research-Open/calisim/tree/main/CONTRIBUTING.md).

# License

calisim is published under the Apache License (see [LICENSE](https://github.com/Plant-Food-Research-Open/calisim/tree/main/LICENSE)).

View all third party licenses (see [third_party](https://github.com/Plant-Food-Research-Open/calisim/tree/main/third_party))

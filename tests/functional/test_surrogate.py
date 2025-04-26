"""
Functional tests for the surrogate module.

A battery of tests to validate the surrogate modelling procedures.

"""

import pytest
import sklearn.gaussian_process.kernels as kernels

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.statistics import DistanceMetricBase
from calisim.surrogate import (
	SurrogateModelMethod,
	SurrogateModelMethodModel,
)

from ..conftest import get_calibrator


def test_sklearn(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="gp",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(kernel=kernels.RBF()),
	)

	calibrator = get_calibrator(
		SurrogateModelMethod,
		SurrogateModelMethodModel,
		sir_model,
		sir_parameter_spec,
		"sklearn",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


@pytest.mark.torch
def test_gpytorch(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="gp",
		n_samples=100,
		n_iterations=100,
		lr=0.01,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		SurrogateModelMethod,
		SurrogateModelMethodModel,
		sir_model,
		sir_parameter_spec,
		"gpytorch",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None

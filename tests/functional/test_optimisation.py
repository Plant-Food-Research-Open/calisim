"""
Functional tests for the optimisation module.

A battery of tests to validate the optimisation calibration procedures.

"""

import pytest

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.optimisation import OptimisationMethod, OptimisationMethodModel
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator, is_close


def test_optuna(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="tpes",
		directions=["minimize"],
		n_iterations=100,
		method_kwargs=dict(n_startup_trials=50),
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		OptimisationMethod,
		OptimisationMethodModel,
		sir_model,
		sir_parameter_spec,
		"optuna",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_emukit(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		acquisition_func="ei",
		n_init=5,
		n_iterations=10,
		n_samples=100,
		use_shap=True,
		test_size=0.1,
		method_kwargs=dict(noise_var=1e-4),
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		OptimisationMethod,
		OptimisationMethodModel,
		sir_model,
		sir_parameter_spec,
		"emukit",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


@pytest.mark.torch
def test_botorch(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		output_labels=["Lynx"],
		n_init=5,
		n_iterations=20,
		use_saasbo=False,
		n_out=1,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		OptimisationMethod,
		OptimisationMethodModel,
		sir_model,
		sir_parameter_spec,
		"botorch",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_openturns(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="kriging",
		n_init=100,
		n_iterations=25,
		n_out=1,
		method_kwargs=dict(
			basis="constant",
			covariance="SquaredExponential",
			covariance_scale=1,
			covariance_amplitude=1,
		),
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		OptimisationMethod,
		OptimisationMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)

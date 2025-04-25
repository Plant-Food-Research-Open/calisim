"""
Functional tests for the ABC module.

A battery of tests to validate the ABC calibration procedures.

"""

from calisim.base import ExampleModelContainer
from calisim.bayesian import (
	BayesianCalibrationMethod,
	BayesianCalibrationMethodModel,
)
from calisim.data_model import ParameterSpecification
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator, is_close


def test_emcee(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	gaussian_ll_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_iterations=100,
		n_samples=32,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		BayesianCalibrationMethod,
		BayesianCalibrationMethodModel,
		sir_model,
		sir_parameter_spec,
		"emcee",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		gaussian_ll_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_openturns(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	gaussian_ll_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_samples=250,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		BayesianCalibrationMethod,
		BayesianCalibrationMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		gaussian_ll_metric,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)

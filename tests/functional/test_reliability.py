"""
Functional tests for the reliability analysis module.

A battery of tests to validate the reliability analysis procedures.

"""

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.reliability import (
	ReliabilityAnalysisMethod,
	ReliabilityAnalysisMethodModel,
)
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator


def test_openturns_monte_carlo(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="monte_carlo",
		n_samples=300,
		comparison="less_or_equal",
		threshold=20,
		n_jobs=1,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		ReliabilityAnalysisMethod,
		ReliabilityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sampler is not None


def test_openturns_subset(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="subset",
		n_samples=300,
		comparison="less_or_equal",
		threshold=20,
		n_jobs=1,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		ReliabilityAnalysisMethod,
		ReliabilityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sampler is not None


def test_openturns_sobol(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="sobol",
		n_samples=300,
		comparison="less_or_equal",
		threshold=20,
		n_jobs=1,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		ReliabilityAnalysisMethod,
		ReliabilityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sampler is not None

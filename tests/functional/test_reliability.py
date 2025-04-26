"""
Functional tests for the reliability analysis module.

A battery of tests to validate the reliability analysis procedures.

"""

import pytest

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.reliability import (
	ReliabilityAnalysisMethod,
	ReliabilityAnalysisMethodModel,
)
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator


def test_openturns_unsupported_sampler(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="__functional_test__",
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported sampler" in str(exc_info.value)


def test_openturns_unsupported_comparison(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="monte_carlo",
		n_samples=300,
		comparison="__functional_test__",
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported comparison" in str(exc_info.value)


def test_openturns_monte_carlo_less_or_equal(
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


def test_openturns_monte_carlo_less(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="monte_carlo",
		n_samples=300,
		comparison="less",
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


def test_openturns_monte_carlo_greater(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="monte_carlo",
		n_samples=300,
		comparison="greater",
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


def test_openturns_monte_carlo_greater_or_equal(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="monte_carlo",
		n_samples=300,
		comparison="greater_or_equal",
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

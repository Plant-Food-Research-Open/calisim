"""
Functional tests for the sensitivity module.

A battery of tests to validate the sensitivity calibration procedures.

"""

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.sensitivity import (
	SensitivityAnalysisMethod,
	SensitivityAnalysisMethodModel,
)
from calisim.statistics import DistanceMetricBase

from ..conftest import get_calibrator


def test_salib(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="sobol",
		n_samples=128,
		method_kwargs=dict(calc_second_order=True, scramble=True),
		analyze_kwargs=dict(
			calc_second_order=True,
			num_resamples=200,
			conf_level=0.95,
		),
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		SensitivityAnalysisMethod,
		SensitivityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"salib",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sp is not None


def test_openturns_saltelli(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="saltelli",
		order=4,
		n_samples=256,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		SensitivityAnalysisMethod,
		SensitivityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sp is not None


def test_openturns_chaos_sobol(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="chaos_sobol",
		order=4,
		n_samples=256,
		calibration_func_kwargs=dict(t=observed_data.day),
	)

	calibrator = get_calibrator(
		SensitivityAnalysisMethod,
		SensitivityAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.implementation.sp is not None

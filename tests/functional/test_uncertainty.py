"""
Functional tests for the uncertainty analysis module.

A battery of tests to validate the uncertainty analysis procedures.

"""

from calisim.base import ExampleModelContainer
from calisim.data_model import (
	ParameterSpecification,
)
from calisim.statistics import DistanceMetricBase
from calisim.uncertainty import (
	UncertaintyAnalysisMethod,
	UncertaintyAnalysisMethodModel,
)

from ..conftest import get_calibrator


def test_chaospy_linear_least_squares(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="least_squares",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_linear_elastic(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="elastic",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_linear_lasso(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="lasso",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_linear_lasso_lars(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="lasso_lars",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_linear_lars(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="lars",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_linear_ridge(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="linear",
		algorithm="ridge",
		method="sobol",
		order=4,
		n_samples=100,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_quadrature(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="quadrature",
		algorithm="least_squares",
		method="gaussian",
		order=4,
		n_samples=5,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_chaospy_gp(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="gp",
		algorithm="least_squares",
		method="sobol",
		order=4,
		n_samples=5,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"chaospy",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_openturns_functional_chaos(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="functional_chaos",
		order=4,
		n_samples=100,
		test_size=0.1,
		n_out=1,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(
			basis="constant",
			covariance="SquaredExponential",
			covariance_scale=1,
			covariance_amplitude=1,
		),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_openturns_kriging(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		solver="kriging",
		order=4,
		n_samples=100,
		test_size=0.1,
		n_out=1,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(
			basis="constant",
			covariance="SquaredExponential",
			covariance_scale=1,
			covariance_amplitude=1,
		),
	)

	calibrator = get_calibrator(
		UncertaintyAnalysisMethod,
		UncertaintyAnalysisMethodModel,
		sir_model,
		sir_parameter_spec,
		"openturns",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
		l2_norm_metric,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None

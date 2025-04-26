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


def test_sklearn_unsupported_emulator(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="__functional_test__",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported emulator" in str(exc_info.value)


def test_sklearn_gp(
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


def test_sklearn_emukit_gp(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="emukit_gp",
		n_samples=5,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_sklearn_rf(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="rf",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_gb(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="gb",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_lr(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="lr",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_elastic(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="elastic",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_ridge(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="ridge",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_knn(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="knn",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_kr(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="kr",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_linear_svm(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="linear_svm",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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


def test_sklearn_nu_svm(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="nu_svm",
		n_samples=50,
		use_shap=True,
		test_size=0.1,
		flatten_Y=True,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
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

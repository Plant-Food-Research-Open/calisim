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


def test_optuna_unsupported_algorithm(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="__functional_test__",
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported Optuna sampler" in str(exc_info.value)


def test_optuna_tpes(
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


def test_optuna_cmaes(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="cmaes",
		directions=["minimize"],
		n_iterations=100,
		method_kwargs=dict(),
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


def test_optuna_nsga(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="nsga",
		directions=["minimize"],
		n_iterations=100,
		method_kwargs=dict(),
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


def test_optuna_qmc(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="qmc",
		directions=["minimize"],
		n_iterations=100,
		method_kwargs=dict(),
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


def test_optuna_gp(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="gp",
		directions=["minimize"],
		n_iterations=100,
		method_kwargs=dict(),
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


def test_emukit_unsupported_acquisition(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		acquisition_func="__functional_test__",
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported acquisition function" in str(exc_info.value)


def test_emukit_ei(
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


def test_emukit_poi(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		acquisition_func="poi",
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


def test_emukit_lp(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		acquisition_func="lp",
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


def test_emukit_nlcb(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		directions=["minimize"],
		acquisition_func="nlcb",
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


def test_openturns_unsupported_algorithm(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
	l2_norm_metric: DistanceMetricBase,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		method="__functional_test__",
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

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported optimisation algorithm" in str(exc_info.value)


def test_openturns_kriging(
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

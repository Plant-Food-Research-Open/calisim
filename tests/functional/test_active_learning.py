"""
Functional tests for the active learning module.

A battery of tests to validate the active learning calibration procedures.

"""

import pytest

from calisim.active_learning import (
	ActiveLearningMethod,
	ActiveLearningMethodModel,
)
from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification

from ..conftest import get_calibrator


def test_skactiveml_unsupported_surrogate(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="__functional_test__",
		query_strategy="greedy_sampling_target",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(alpha=1e-10, optimizer="fmin_l_bfgs_b"),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported surrogate class" in str(exc_info.value)


def test_skactiveml_unsupported_query_strategy(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="gp",
		query_strategy="__functional_test__",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(alpha=1e-10, optimizer="fmin_l_bfgs_b"),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	with pytest.raises(ValueError) as exc_info:
		calibrator.specify().execute().analyze()
	assert "Unsupported query strategy" in str(exc_info.value)


def test_skactiveml_gp_greedy_sampling_target(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="gp",
		query_strategy="greedy_sampling_target",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(alpha=1e-10, optimizer="fmin_l_bfgs_b"),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_skactiveml_rf_greedy_sampling_targets(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="rf",
		query_strategy="greedy_sampling_target",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(n_estimators=100),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_skactiveml_nick_expected_model_variance_reduction(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="nick",
		query_strategy="expected_model_variance_reduction",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None


def test_skactiveml_dt_regression_tree_based_al(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_init=20,
		n_iterations=10,
		n_samples=50,
		lr=0.01,
		use_shap=True,
		method="dt",
		query_strategy="regression_tree_based_al",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
	)

	calibrator = get_calibrator(
		ActiveLearningMethod,
		ActiveLearningMethodModel,
		sir_model,
		sir_parameter_spec,
		"skactiveml",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert calibrator.get_emulator() is not None

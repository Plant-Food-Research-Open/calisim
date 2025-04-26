"""
Functional tests for the history matching module.

A battery of tests to validate the history matching calibration procedures.

"""

import numpy as np

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.history_matching import HistoryMatchingMethod, HistoryMatchingMethodModel

from ..conftest import get_calibrator, is_close


def test_ies_sies(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data
	output_labels = sir_model.output_labels

	calibration_kwargs = dict(
		method="sies",
		n_samples=50,
		n_iterations=10,
		covariance=np.eye(observed_data[output_labels].values.flatten().shape[0]),
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(truncation=1.0),
	)

	calibrator = get_calibrator(
		HistoryMatchingMethod,
		HistoryMatchingMethodModel,
		sir_model,
		sir_parameter_spec,
		"ies",
		outdir,
		output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_ies_esmda(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data
	output_labels = sir_model.output_labels

	calibration_kwargs = dict(
		method="esmda",
		n_samples=50,
		n_iterations=10,
		covariance=np.eye(observed_data[output_labels].values.flatten().shape[0]),
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(),
	)

	calibrator = get_calibrator(
		HistoryMatchingMethod,
		HistoryMatchingMethodModel,
		sir_model,
		sir_parameter_spec,
		"ies",
		outdir,
		output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_pyesmda_esmda_rs(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data
	output_labels = sir_model.output_labels

	calibration_kwargs = dict(
		method="esmda_rs",
		n_samples=50,
		n_iterations=20,
		covariance=np.eye(observed_data[output_labels].values.flatten().shape[0]),
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(save_ensembles_history=True),
		n_jobs=10,
	)

	calibrator = get_calibrator(
		HistoryMatchingMethod,
		HistoryMatchingMethodModel,
		sir_model,
		sir_parameter_spec,
		"pyesmda",
		outdir,
		output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)


def test_pyesmda_esmda(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data
	output_labels = sir_model.output_labels

	calibration_kwargs = dict(
		method="esmda",
		n_samples=50,
		n_iterations=20,
		covariance=np.eye(observed_data[output_labels].values.flatten().shape[0]),
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(save_ensembles_history=True),
		n_jobs=10,
	)

	calibrator = get_calibrator(
		HistoryMatchingMethod,
		HistoryMatchingMethodModel,
		sir_model,
		sir_parameter_spec,
		"pyesmda",
		outdir,
		output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)

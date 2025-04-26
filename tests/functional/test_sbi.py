"""
Functional tests for the simulation-based inference module.

A battery of tests to validate the simulation-based inference procedures.

"""

import pytest

from calisim.base import ExampleModelContainer
from calisim.data_model import ParameterSpecification
from calisim.sbi import (
	SimulationBasedInferenceMethod,
	SimulationBasedInferenceMethodModel,
)

from ..conftest import get_calibrator, is_close


@pytest.mark.torch
def test_lampe(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_samples=300,
		n_iterations=25,
		num_simulations=200,
		lr=0.001,
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(transforms=20, hidden_features=[64] * 3),
	)

	calibrator = get_calibrator(
		SimulationBasedInferenceMethod,
		SimulationBasedInferenceMethodModel,
		sir_model,
		sir_parameter_spec,
		"lampe",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator, rtol=0.5)


@pytest.mark.torch
def test_sbi(
	sir_model: ExampleModelContainer,
	sir_parameter_spec: ParameterSpecification,
	outdir: str,
) -> None:
	observed_data = sir_model.observed_data

	calibration_kwargs = dict(
		n_samples=100,
		n_iterations=100,
		num_simulations=100,
		method="nsf",
		calibration_func_kwargs=dict(t=observed_data.day),
		method_kwargs=dict(
			hidden_features=40,
			num_transforms=40,
		),
	)

	calibrator = get_calibrator(
		SimulationBasedInferenceMethod,
		SimulationBasedInferenceMethodModel,
		sir_model,
		sir_parameter_spec,
		"sbi",
		outdir,
		sir_model.output_labels,
		calibration_kwargs,
	)

	calibrator.specify().execute().analyze()
	assert is_close(sir_model, calibrator)

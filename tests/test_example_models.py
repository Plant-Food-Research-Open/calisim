"""
Tests for various example models.

A battery of tests to validate the example models.

"""

import numpy as np

from calisim.base import ExampleModelContainer
from calisim.example_models import (
	AnharmonicOscillator,
	Lorenz95,
	LotkaVolterraModel,
	SirOdesModel,
)


def test_lotka_volterra_runs() -> None:
	model = LotkaVolterraModel()
	container = ExampleModelContainer(model)
	output_labels = container.output_labels
	observed_data = container.observed_data

	simulation_parameters = container.ground_truth.copy()
	simulation_parameters["t"] = observed_data.year

	simulated_data = model.simulate(simulation_parameters)[
		output_labels
	].values.flatten()

	is_close_comparison = np.isclose(
		simulated_data, observed_data[output_labels].values.flatten(), rtol=1
	)
	assert np.all(is_close_comparison)


def test_sir_odes_runs() -> None:
	model = SirOdesModel()
	container = ExampleModelContainer(model)
	output_labels = container.output_labels
	observed_data = container.observed_data

	simulation_parameters = container.ground_truth.copy()
	simulation_parameters["t"] = observed_data.day

	simulated_data = model.simulate(simulation_parameters)[
		output_labels
	].values.flatten()

	is_close_comparison = np.isclose(
		simulated_data, observed_data[output_labels].values.flatten(), rtol=1
	)
	assert np.all(is_close_comparison)


def test_anharmonic_oscillator_runs() -> None:
	model = AnharmonicOscillator()
	container = ExampleModelContainer(model)
	output_labels = container.output_labels
	observed_data = container.observed_data

	simulation_parameters = container.ground_truth.copy()
	simulation_parameters["t"] = observed_data.index

	simulated_data = model.simulate(simulation_parameters)[
		output_labels
	].values.flatten()

	is_close_comparison = np.isclose(
		simulated_data, observed_data[output_labels].values.flatten(), rtol=1
	)
	assert np.all(is_close_comparison)


def test_lorenz_runs() -> None:
	model = Lorenz95()
	container = ExampleModelContainer(model)
	simulation_parameters = container.ground_truth.copy()
	simulated_data = model.simulate(simulation_parameters).values.flatten()
	assert simulated_data is not None

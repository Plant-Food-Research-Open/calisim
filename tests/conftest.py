"""
Defines fixtures for the testing system.

A collection of fixtures for reuse across all tests.

"""

import os.path as osp
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from calisim.base import CalibrationMethodBase, ExampleModelContainer
from calisim.data_model import (
	CalibrationModel,
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import SirOdeModel
from calisim.statistics import DistanceMetricBase, GaussianLogLikelihood, L2Norm


def pytest_addoption(parser: pytest.Parser) -> None:
	"""Add options to the PyTest command parser.

	Args:
	    parser (pytest.Parser): The command parser.
	"""
	parser.addoption(
		"--torch", action="store_true", default=False, help="Run PyTorch tests"
	)


def pytest_configure(config: pytest.Config) -> None:
	"""Add value the PyTest configuration.

	Args:
	    config (pytest.Config): The PyTest configuration.
	"""
	config.addinivalue_line("markers", "torch: Run PyTorch tests")


def pytest_collection_modifyitems(
	config: pytest.Config, items: list[pytest.Item]
) -> None:
	"""Modify the PyTest configuration item list.

	Args:
	    config (pytest.Config): The PyTest configuration.
	    items (list[pytest.Item]): The item list.
	"""
	if config.getoption("--torch"):
		return
	skip_torch = pytest.mark.skip(reason="Need --torch option to run")
	for item in items:
		if "torch" in item.keywords:
			item.add_marker(skip_torch)


@pytest.fixture
def sir_model() -> ExampleModelContainer:
	"""Get the SIR model.

	Returns:
	    SirOdeModel: The SIR model.
	"""
	model = SirOdeModel()
	container = ExampleModelContainer(model)
	return container


@pytest.fixture
def sir_parameter_spec() -> ParameterSpecification:
	"""Get the SIR parameter specification.

	Returns:
	    ParameterSpecification: The SIR parameter specification.
	"""
	parameter_spec = ParameterSpecification(
		parameters=[
			DistributionModel(
				name="gamma",
				distribution_name="uniform",
				distribution_args=[0.09, 0.11],
				distribution_bounds=[0.05, 0.15],
				data_type=ParameterDataType.CONTINUOUS,
			),
			DistributionModel(
				name="beta",
				distribution_name="uniform",
				distribution_args=[0.39, 0.41],
				distribution_bounds=[0.3, 0.5],
				data_type=ParameterDataType.CONTINUOUS,
			),
		]
	)
	return parameter_spec


@pytest.fixture
def outdir() -> str:
	"""Get the output directory.

	Returns:
	    str: The output directory.
	"""
	return osp.join("tests", "outdir")


@pytest.fixture
def l2_norm_metric() -> DistanceMetricBase:
	"""Get the L2 norm distance metric.

	Returns:
	    DistanceMetricBase: The L2 norm distance metric instance.
	"""
	return L2Norm()


@pytest.fixture
def gaussian_ll_metric() -> DistanceMetricBase:
	"""Get the Gaussian log likelihood metric.

	Returns:
	    DistanceMetricBase: The Gaussian log likelihood metric instance.
	"""
	return GaussianLogLikelihood()


def get_parameter_estimates(calibrator: CalibrationMethodBase) -> dict[str, float]:
	"""Get flattened parameter estimates.

	Args:
	    calibrator (CalibrationMethodBase): The simulation calibrator.

	Returns:
	    dict[str, float]: The flattened parameter estimates.
	"""
	parameter_estimates = {
		estimate.name: estimate.estimate
		for estimate in calibrator.get_parameter_estimates().estimates
	}
	return parameter_estimates


def is_close(
	model: ExampleModelContainer, calibrator: CalibrationMethodBase, rtol: float = 0.3
) -> bool:
	"""Check if all parameter estimates are close to the ground truth.

	Args:
	    model (ExampleModelContainer): The model container.
	    calibrator (CalibrationMethodBase): The model calibrator.
	    rtol (float, optional): The relative tolerance. Defaults to 0.15.

	Returns:
	    bool: Whether all parameter estimates are close to the true values.
	"""
	parameter_estimates = get_parameter_estimates(calibrator)
	ground_truth = np.array([model.ground_truth[k] for k in parameter_estimates])
	estimate_values = np.array([parameter_estimates[k] for k in parameter_estimates])

	is_close_comparison = np.isclose(ground_truth, estimate_values, rtol=rtol)
	return np.all(is_close_comparison)


def get_calibration_func(
	model_container: ExampleModelContainer,
	output_labels: list[str],
	metric: DistanceMetricBase | None = None,
) -> Callable:
	"""Get the function for the SIR calibration procedure.

	Args:
	    model_container (ExampleModelContainer): The simulation model container.
	    output_labels (list[str]): The collection of output labels.
	    metric (DistanceMetricBase | None, optional): The discrepancy metric.
	        Defaults to None.

	Returns:
	    Callable: The calibration function.
	"""
	model = model_container.model
	ground_truth = model_container.ground_truth

	def calibration_func(
		parameters: dict,
		simulation_id: str,
		observed_data: np.ndarray | None,
		t: pd.Series,
	) -> float | list[float]:
		simulation_parameters = ground_truth.copy()
		simulation_parameters["t"] = t

		for k in parameters:
			simulation_parameters[k] = parameters[k]

		simulated_data = model.simulate(simulation_parameters)[
			output_labels
		].values.flatten()

		if metric is None:
			return simulated_data
		else:
			discrepancy = metric.calculate(observed_data, simulated_data)
			return discrepancy

	return calibration_func


def get_calibrator(
	calibrator_type: type[CalibrationMethodBase],
	spec_type: type[CalibrationModel],
	model_container: ExampleModelContainer,
	parameter_spec: ParameterSpecification,
	engine: str,
	outdir: str,
	output_labels: list[str],
	calibration_kwargs: dict[str, Any],
	metric: DistanceMetricBase | None = None,
) -> CalibrationMethodBase:
	"""Get the simulation calibrator.

	Args:
	    calibrator_type (type[CalibrationMethodBase]): The simulation calibrator type.
	    spec_type (type[CalibrationModel]): The calibration specification type.
	    model_container (ExampleModelContainer): The simulation model container.
	    parameter_spec (ParameterSpecification): The simulation parameter specification.
	    engine (str): The calibration engine.
	    outdir (str): The calibration output directory.
	    output_labels (list[str]): The simulation output labels.
	    calibration_kwargs (dict[str, Any]): The calibration named arguments.
	    metric (DistanceMetricBase | None, optional): The discrepancy metric.
	        Defaults to None.

	Returns:
	    CalibrationMethodBase: The calibration method.
	"""
	observed_data = model_container.observed_data

	calibration_func = get_calibration_func(model_container, output_labels, metric)

	specification = spec_type(
		experiment_name=f"{engine}_functional_test",
		parameter_spec=parameter_spec,
		observed_data=observed_data[output_labels].values.flatten(),
		outdir=outdir,
		verbose=True,
		batched=False,
		random_seed=100,
		**calibration_kwargs,
	)

	calibrator = calibrator_type(
		calibration_func=calibration_func, specification=specification, engine=engine
	)  # type: ignore [call-arg]
	return calibrator

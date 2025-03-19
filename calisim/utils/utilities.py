"""Contains utility functions for calibration

This module defines various utility functions
for the calibration of simulations.

"""

import os.path as osp
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_model import ParameterDataType


def get_datetime_now() -> str:
	"""Get the current datetime for now.

	Returns:
	    str: The current datetime.
	"""
	return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def get_simulation_uuid() -> str:
	"""Get a new simulation uuid.

	Returns:
	    str: The simulation uuid.
	"""
	simulation_uuid = str(uuid.uuid4())
	return simulation_uuid


def get_examples_outdir() -> str:
	"""Get the output directory for calibration examples.

	Returns:
	    str: The output directory.
	"""
	return osp.join("examples", "outdir")


def create_file_path(file_path: str) -> str:
	"""Create file path if it does not exist.

	Args:
		file_path (str): The file path to create.

	Returns:
		str: The created file path.
	"""
	path = Path(file_path)
	if not path.is_dir():
		path.mkdir(parents=True, exist_ok=True)
	return file_path


def calibration_func_wrapper(
	X: np.ndarray,
	workflow: "CalibrationWorkflowBase",  # type: ignore[name-defined] # noqa: F821
	observed_data: pd.DataFrame | np.ndarray,
	parameter_names: list[str],
	data_types: list[ParameterDataType],
	calibration_kwargs: dict,
	wrap_values: bool = False,
) -> np.ndarray:
	"""Wrapper function for the calibration function.

	Args:
		X (np.ndarray): The parameter set matrix.
		workflow (CalibrationWorkflowBase): The calibration workflow.
		observed_data (pd.DataFrame | np.ndarray): The observed data.
		parameter_names (list[str]): The list of simulation parameter names.
		data_types (list[ParameterDataType]): The data types for each parameter.
		calibration_kwargs (dict): Arguments to supply to the calibration function.
		wrap_values (bool): Whether to wrap scalar values with a list.
			Defaults to False.

	Returns:
		np.ndarray: The simulation output data.
	"""
	import numpy as np

	parameters = []
	for theta in X:
		parameter_set = {}
		for i, parameter_value in enumerate(theta):
			parameter_name = parameter_names[i]
			data_type = data_types[i]
			if data_type == ParameterDataType.CONTINUOUS:
				parameter_set[parameter_name] = parameter_value
			else:
				parameter_set[parameter_name] = int(parameter_value)
		parameters.append(parameter_set)

	simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]

	if workflow.specification.batched:
		results = workflow.call_calibration_func(
			parameters, simulation_ids, observed_data, **calibration_kwargs
		)
	else:
		results = []
		for i, parameter in enumerate(parameters):
			simulation_id = simulation_ids[i]
			result = workflow.call_calibration_func(
				parameter,
				simulation_id,
				observed_data,
				**calibration_kwargs,
			)
			if wrap_values and not isinstance(result, list):
				result = [result]
			results.append(result)
	results = np.array(results)
	return results


class EarlyStopper:
	"""Early stopping for training."""

	def __init__(self, patience: int = 1, min_delta: float = 0):
		"""EarlyStopper constructor.

		Args:
		    patience (int, optional): The number of iterations
		           before performing early stopping. Defaults to 1.
		    min_delta (float, optional): The minimum difference
		       for the validation loss. Defaults to 0.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter: int = 0
		self.min_validation_loss = float("inf")

	def early_stop(self, validation_loss: float) -> bool:
		"""Perform early stopping.

		Args:
		    validation_loss (float): The training validation loss.

		Returns:
		    bool: Whether to perform early stopping.
		"""
		if validation_loss < self.min_validation_loss:
			self.min_validation_loss = validation_loss
			self.counter = 0
		elif validation_loss > (self.min_validation_loss + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


def extend_X(X: np.ndarray, Y_rows: int) -> np.ndarray:
	"""Extend the number of rows for X with a dummy index column.

	Args:
		X (np.ndarray): The input matrix.
		Y_rows (int) The number of rows for the simulation outputs.

	Returns:
		np.ndarray: The extended input matrix with a dummy column.
	"""
	design_list = []
	for i in range(X.shape[0]):
		for j in range(Y_rows):
			row = np.append(X[i], j)
			design_list.append(row)
	X = np.array(design_list)
	return X

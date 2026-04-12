"""Contains the Emukit base class

The defined base class for the Emukit library.

"""

from collections.abc import Callable

import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign

from ..data_model import ParameterDataType
from .calibration_base import CalibrationWorkflowBase


class EmukitBase(CalibrationWorkflowBase):
	"""The Emukit base class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		parameters = []
		self.names = []
		self.data_types = []
		self.bounds = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			data_type = spec.data_type

			if data_type == ParameterDataType.DISCRETE:
				bounds = self.get_parameter_bounds(spec)
				self.bounds.append(bounds)
				lower_bound, upper_bound = bounds
				discrete_domain = np.arange(lower_bound, upper_bound + 1)
				parameter = DiscreteParameter(parameter_name, discrete_domain)
			elif data_type == ParameterDataType.CONSTANT:
				parameter_value = spec.parameter_value
				self.constants[parameter_name] = parameter_value
				continue
			elif data_type == ParameterDataType.CATEGORICAL:
				pass
			else:
				bounds = self.get_parameter_bounds(spec)
				self.bounds.append(bounds)
				lower_bound, upper_bound = bounds
				parameter = ContinuousParameter(
					parameter_name, lower_bound, upper_bound
				)
			self.names.append(parameter_name)
			self.data_types.append(data_type)
			parameters.append(parameter)  # type: ignore[possibly-undefined]

		self.parameters = ParameterSpace(parameters)

	def get_X_Y(
		self, n_init: int, target_function: Callable
	) -> tuple[np.ndarray, np.ndarray]:
		"""Get the X and Y matrices.

		Args:
			n_init (int): The number of samples to take
				from the random design.
			target_function (Callable):
				The simulation function.

		Returns:
			tuple[np.ndarray, np.ndarray]: The X and Y matrices.
		"""
		design = RandomDesign(self.parameters)
		X = self.specification.X
		if X is None:
			X = design.get_samples(n_init)

		n_replicates = self.specification.n_replicates
		if n_replicates > 1:
			X = np.repeat(X, n_replicates, axis=0)
			self.rng.shuffle(X)

		Y = self.specification.Y
		if Y is None:
			Y = target_function(X)
		return X, Y

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Get new parameter samples.

		Args:
		    n_samples (int): The number of samples.

		Returns:
		    np.ndarray: The parameter samples.
		"""
		design = RandomDesign(self.parameters)
		X = design.get_samples(n_samples)
		return X

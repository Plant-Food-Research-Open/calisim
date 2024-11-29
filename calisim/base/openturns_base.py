"""Contains the OpenTurns base class

The defined base class for performing OpenTurns.

"""

from collections.abc import Callable

import numpy as np
import openturns as ot

from .calibration_base import CalibrationWorkflowBase


class OpenTurnsBase(CalibrationWorkflowBase):
	"""The OpenTurns base class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.bounds: tuple[list[float], list[float]] = ([], [])
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			bounds = self.get_parameter_bounds(spec)
			lower_bound, upper_bound = bounds
			lower_bounds, upper_bounds = self.bounds
			lower_bounds.append(lower_bound)
			upper_bounds.append(upper_bound)

			data_type = spec.data_type
			self.data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			dist_instance = getattr(ot, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters.append(parameter)

		distribution_collection = ot.DistributionCollection(parameters)
		self.parameters = ot.JointDistribution(distribution_collection)

	def get_ot_func_wrapper(self, func_sample: Callable) -> Callable:
		"""Get an OpenTurns wrapper for Python functions.

		Args:
		    func_sample (Callable): The function to wrap.

		Returns:
		    Callable: The wrapped function.
		"""
		n_dim = self.parameters.getDimension()
		n_out = self.specification.n_out

		return ot.PythonFunction(n_dim, n_out, func_sample=func_sample)

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Get new parameter samples.

		Args:
		    n_samples (int): The number of samples.

		Returns:
		    np.ndarray: The parameter samples.
		"""
		if not hasattr(self, "experiment"):
			self.experiment = ot.LHSExperiment(self.parameters, n_samples)

		self.experiment.setSize(n_samples)
		samples = self.experiment.generate()
		return samples

	def get_X_Y(
		self, n_samples: int, target_function: Callable
	) -> tuple[np.ndarray, np.ndarray]:
		"""Get the X and Y matrices.

		Args:
		    n_samples (int): The number of samples to take
		        from the random design.
		    target_function (Callable):
		        The simulation function.

		Returns:
		    tuple[np.ndarray, np.ndarray]: The X and Y matrices.
		"""
		X = self.specification.X
		if X is None:
			X = self.sample_parameters(n_samples)

		Y = self.specification.Y
		if Y is None:
			ot_func_wrapper = self.get_ot_func_wrapper(target_function)
			Y = ot_func_wrapper(X)

		return X, Y

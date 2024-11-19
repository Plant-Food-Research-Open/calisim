"""Contains the surrogate modelling base class

The defined base class for surrogate modelling.

"""

import chaospy
import numpy as np

from .calibration_base import CalibrationWorkflowBase


class SurrogateBase(CalibrationWorkflowBase):
	"""The surrogate modelling base class."""

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Sample from the paramete space.

		Args:
			n_samples (int): The number of samples.

		Returns:
			np.ndarray: The sampled parameter values.
		"""
		return self.parameters.sample(n_samples, rule="sobol").T

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

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

			dist_instance = getattr(chaospy, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters.append(parameter)

		self.parameters = chaospy.J(*parameters)

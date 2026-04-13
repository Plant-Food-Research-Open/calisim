"""Contains the surrogate modelling base class

The defined base class for surrogate modelling.

"""

import chaospy
import numpy as np

from ..data_model import ParameterDataType
from .calibration_base import CalibrationWorkflowBase


class SurrogateBase(CalibrationWorkflowBase):
	"""The surrogate modelling base class."""

	def sample_parameters(self, n_samples: int) -> np.ndarray:
		"""Sample from the parameter space.

		Args:
			n_samples (int): The number of samples.

		Returns:
			np.ndarray: The sampled parameter values.
		"""
		X = self.parameters.sample(n_samples, rule="sobol").T
		n_replicates = self.specification.n_replicates
		if n_replicates > 1:
			X = np.repeat(X, n_replicates, axis=0)
			self.rng.shuffle(X)
		return X

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			data_type = spec.data_type

			if data_type == ParameterDataType.CONSTANT:
				parameter_value = spec.parameter_value
				self.constants[parameter_name] = parameter_value
				continue
			elif data_type == ParameterDataType.CATEGORICAL:
				lower_bound, upper_bound = self.set_categorical_parameter(spec)
				parameter = chaospy.DiscreteUniform(lower_bound, upper_bound - 1)
			else:
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

			self.names.append(parameter_name)
			self.data_types.append(data_type)
			parameters.append(parameter)

		self.parameters = chaospy.J(*parameters)

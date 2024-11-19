"""Contains the Emukit base class

The defined base class for the Emukit library.

"""

import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace

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
			self.names.append(parameter_name)

			bounds = self.get_parameter_bounds(spec)
			self.bounds.append(bounds)
			lower_bound, upper_bound = bounds

			data_type = spec.data_type
			self.data_types.append(data_type)

			if data_type == ParameterDataType.DISCRETE:
				discrete_domain = np.arange(lower_bound, upper_bound + 1)
				parameter = DiscreteParameter(parameter_name, discrete_domain)
			else:
				parameter = ContinuousParameter(
					parameter_name, lower_bound, upper_bound
				)
			parameters.append(parameter)

		self.parameter_space = ParameterSpace(parameters)

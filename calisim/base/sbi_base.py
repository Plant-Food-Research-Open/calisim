"""Contains the simulation-based inference base class

The defined base class for performing simulation-based inference.

"""

import numpy as np
import torch
import torch.distributions as dist

from ..data_model import ParameterDataType
from .calibration_base import CalibrationWorkflowBase


class SimulationBasedInferenceBase(CalibrationWorkflowBase):
	"""The simulation-based inference base class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			name = spec.name
			self.names.append(name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			if data_type == ParameterDataType.DISCRETE:
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				lower_bound = np.floor(lower_bound).astype("int")
				upper_bound = np.floor(upper_bound).astype("int")
				replicates = np.floor(upper_bound - lower_bound).astype("int")
				probabilities = torch.tensor([1 / replicates])
				probabilities = probabilities.repeat(replicates)
				base_distribution = dist.Categorical(probabilities)
				transforms = [
					dist.AffineTransform(
						loc=torch.Tensor([lower_bound]), scale=torch.Tensor([1])
					)
				]
				prior = dist.TransformedDistribution(base_distribution, transforms)
			else:
				distribution_name = (
					spec.distribution_name.replace("_", " ").title().replace(" ", "")
				)

				distribution_args = spec.distribution_args
				if distribution_args is None:
					distribution_args = []
				distribution_args = [torch.Tensor([arg]) for arg in distribution_args]

				distribution_kwargs = spec.distribution_kwargs
				if distribution_kwargs is None:
					distribution_kwargs = {}
				distribution_kwargs = {
					k: torch.Tensor([v]) for k, v in distribution_kwargs.items()
				}

				distribution_class = getattr(dist, distribution_name)

				prior = distribution_class(*distribution_args, **distribution_kwargs)

			self.parameters.append(prior)

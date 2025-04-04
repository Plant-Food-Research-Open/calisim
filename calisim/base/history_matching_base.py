"""Contains the history matching base class

The defined base class for performing history matching.

"""

import numpy as np

from .calibration_base import CalibrationWorkflowBase


class HistoryMatchingBase(CalibrationWorkflowBase):
	"""The history matching base class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		ensemble_size = self.specification.n_samples
		n_replicates = self.specification.n_replicates
		parameter_spec = self.specification.parameter_spec.parameters

		self.parameters = {}
		for spec in parameter_spec:
			parameter_name = spec.name
			distribution_name = spec.distribution_name.replace(" ", "_").lower()

			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}
			distribution_kwargs["size"] = ensemble_size

			dist_instance = getattr(self.rng, distribution_name)
			samples = dist_instance(*distribution_args, **distribution_kwargs)
			samples = np.repeat(samples, n_replicates)
			self.rng.shuffle(samples)
			self.parameters[parameter_name] = samples

		self.specification.n_samples = ensemble_size * n_replicates

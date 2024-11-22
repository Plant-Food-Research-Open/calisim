"""Contains the history matching base class

The defined base class for performing history matching.

"""

from .calibration_base import CalibrationWorkflowBase


class HistoryMatchingBase(CalibrationWorkflowBase):
	"""The history matching base class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		ensemble_size = self.specification.n_samples
		parameter_spec = self.specification.parameter_spec.parameters
		self.rng = self.get_default_rng(self.specification.random_seed)

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
			self.parameters[parameter_name] = dist_instance(
				*distribution_args, **distribution_kwargs
			)

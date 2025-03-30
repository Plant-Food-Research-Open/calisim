"""Contains the ELFI base class

The defined base class for the ELFI library.

"""

from collections.abc import Callable

import elfi
import numpy as np

from ..base import CalibrationWorkflowBase


class ELFIBase(CalibrationWorkflowBase):
	"""The ELFI base class."""

	def dist_name_processing(self, name: str) -> str:
		"""Apply data preprocessing to the distribution name.

		Args:
			name (str): The unprocessed distribution name.

		Returns:
			str: The processed distribution name.
		"""
		name = name.replace(" ", "_").lower()

		if name == "normal":
			name = "norm"
		return name

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.bounds = {}
		self.priors = []
		parameter_spec = self.specification.parameter_spec.parameters

		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			distribution_name = self.dist_name_processing(spec.distribution_name)

			bounds = spec.distribution_bounds
			if bounds is not None:
				self.bounds[parameter_name] = bounds

			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			prior = elfi.Prior(
				distribution_name,
				*distribution_args,
				**distribution_kwargs,
				name=parameter_name,
			)
			self.priors.append(prior)

	def create_simulator(
		self, simulator_func: Callable, take_log: bool = False
	) -> None:
		"""Create a simulator object.

		Args:
			simulator_func (Callable): The simulator function.
			take_log (bool, optional): Whether to take the log distance.
				Defaults to False.
		"""
		simulator = elfi.Simulator(
			simulator_func,
			*self.priors,
			observed=self.specification.observed_data,
			name="simulator",
		)
		identity_statistic = elfi.Summary(
			lambda y: y, simulator, name="identity_statistic"
		)

		distance = elfi.Distance(
			lambda simulated, _: simulated, identity_statistic, name="identity_distance"
		)
		if take_log:
			distance = elfi.Operation(np.log, distance, name="log_distance")

		self.distance = distance

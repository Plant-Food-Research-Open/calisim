"""Contains base classes for various example models

Abstract base classes are defined for the
example simulation models.

"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ExampleModelBase(ABC):
	"""The example simulation model abstract class."""

	@abstractmethod
	def get_observed_data(self) -> np.ndarray | pd.DataFrame:
		"""Retrieve observed data.

		Raises:
		    NotImplementedError: Error raised for
				the unimplemented abstract method.

		Returns:
		    np.ndarray | pd.DataFrame: The observed data.
		"""
		raise NotImplementedError("get_observed_data() method not implemented.")

	@abstractmethod
	def simulate(self, parameters: dict) -> np.ndarray | pd.DataFrame:
		"""Run the simulation.

			Args:
		        parameters (dict): The simulation parameters.

		    Raises:
				NotImplementedError: Error raised for the
					unimplemented abstract method.

		Returns:
		    np.ndarray | pd.DataFrame: The simulated data.
		"""
		raise NotImplementedError("simulate() method not implemented.")

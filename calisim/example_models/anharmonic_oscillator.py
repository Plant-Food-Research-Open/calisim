"""Contains an implementation of the anharmonic oscillator model.

An implementation of an anharmonic oscillator example model.

"""

import numpy as np
import pandas as pd

from ..base import ExampleModelBase

# Refer to https://iterative-ensemble-smoother.readthedocs.io/en/latest/Oscillator.html


class AnharmonicOscillator(ExampleModelBase):
	"""Anharmonic oscillator simulation model."""

	def __init__(self) -> None:
		"""AnharmonicOscillator constructor."""
		super().__init__()

		self.reset_y()

		self.GROUND_TRUTH = {"omega": 3.5e-2, "lambda": 3e-4, "K": 2000}
		self.OUTPUT_LABELS = ["y"]

	def reset_y(self) -> None:
		"""Reset the trajectory vector."""
		self.y = [0, 1]

	def get_observed_data(self) -> np.ndarray | pd.DataFrame:
		"""Retrieve observed data.

		Returns:
		    np.ndarray | pd.DataFrame: The observed data.
		"""
		observed_df = self.simulate(self.GROUND_TRUTH)
		observed_df["t"] = observed_df.index
		return observed_df

	def get_df(self) -> pd.DataFrame:
		"""Get the simulation dataframe.

		Returns:
			pd.DataFrame: The simulation dataframe.
		"""
		return pd.DataFrame(dict(y=self.y))

	def simulate(self, parameters: dict) -> np.ndarray | pd.DataFrame:
		"""Run the simulation.

		Args:
			parameters (dict): The simulation parameters.

		Returns:
		    np.ndarray | pd.DataFrame: The simulated data.
		"""
		self.reset_y()
		K = parameters["K"]
		for _ in range(len(self.y), K):
			self.step(parameters)

		df = self.get_df()
		return df

	def step(self, parameters: dict) -> None:
		"""Run the model for one time step.

		Args:
			parameters (dict): The simulation parameters.
		"""
		y_k = (
			self.y[-1]
			* (
				2
				+ parameters["omega"] ** 2
				- parameters["lambda"] ** 2 * self.y[-1] ** 2
			)
			- self.y[-2]
		)

		self.y.append(y_k)

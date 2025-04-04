"""Contains an implementation of a Lorenz95 model.

An implementation of a Lorenz95 example model.

"""

import numpy as np
import pandas as pd

from ..base import ExampleModelBase


class Lorenz95(ExampleModelBase):
	"""The Lorenz95 model."""

	def __init__(self) -> None:
		"""Lorenz95 constructor."""
		super().__init__()

		self.GROUND_TRUTH = dict(F=5, steps=1000, dt=0.01, N=40, noise_std=0)

	def lorenz95(self, x: np.ndarray, F: float) -> np.ndarray:
		"""The Lorenz 95 model.

		Args:
		    x (np.ndarray): The input data.
		    F (float): The force.

		Returns:
		    np.ndarray: The simulation output.
		"""
		N = len(x)
		dxdt = []
		for i in range(N):
			y = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
			dxdt.append(y)
		return np.array(dxdt)

	def rk4_step(self, x: np.ndarray, F: float, dt: float) -> np.ndarray:
		"""Runge-Kutta 4th order integration.

		Args:
		    x (np.ndarray): The input data.
		    F (float): The force.
		    dt (float): Rate of change over time.

		Returns:
		    np.ndarray: The integrated values.
		"""
		k1 = self.lorenz95(x, F)
		k2 = self.lorenz95(x + 0.5 * dt * k1, F)
		k3 = self.lorenz95(x + 0.5 * dt * k2, F)
		k4 = self.lorenz95(x + dt * k3, F)
		return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

	def get_observed_data(self) -> np.ndarray | pd.DataFrame:
		"""Retrieve observed data.

		Returns:
		    np.ndarray | pd.DataFrame: The observed data.
		"""
		observed_df = self.simulate(self.GROUND_TRUTH)
		return observed_df

	def simulate(self, parameters: dict) -> np.ndarray | pd.DataFrame:
		"""Run the simulation.

		Args:
		    parameters (dict): The simulation parameters.

		Returns:
		    np.ndarray | pd.DataFrame: The simulated data.
		"""
		N = parameters["N"]
		F = parameters["F"]
		steps = parameters["steps"]

		x = F * np.ones(N)
		x[0] += 1e-7
		trajectory = [x.copy()]

		for _ in range(steps):
			x = self.step(parameters, x)
			trajectory.append(x.copy())

		trajectory = np.array(trajectory)
		df = pd.DataFrame(trajectory)
		return df

	def step(self, parameters: dict, x: np.ndarray | None = None) -> np.ndarray:
		dt = parameters["dt"]
		F = parameters["F"]
		N = parameters["N"]
		noise_std = np.exp(parameters["noise_std"])

		if x is None:
			x = F * np.ones(N)
			x[0] += 1e-7

		rng = np.random.default_rng()
		noise = rng.normal(0, noise_std, size=N)
		x = self.rk4_step(x, F, dt) + noise
		return x

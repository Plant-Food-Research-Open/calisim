"""Contains an implementation of the Lotka Volterra model.

An implementation of a Lotka Volterra example model.

"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from ..base import ExampleModelBase

# Refer to https://www.pymc.io/projects/examples/en/latest/samplers/SMC-ABC_Lotka-Volterra_example.html


class LotkaVolterraModel(ExampleModelBase):
	"""Lotka Volterra simulation model."""

	def get_observed_data(self) -> np.ndarray | pd.DataFrame:
		"""Retrieve observed data.

		Returns:
		    np.ndarray | pd.DataFrame: The observed data.
		"""
		observed_df = pd.DataFrame(
			dict(
				year=np.arange(1900.0, 1921.0, 1),
				lynx=np.array(
					[
						4.0,
						6.1,
						9.8,
						35.2,
						59.4,
						41.7,
						19.0,
						13.0,
						8.3,
						9.1,
						7.4,
						8.0,
						12.3,
						19.5,
						45.7,
						51.1,
						29.7,
						15.8,
						9.7,
						10.1,
						8.6,
					]
				),
				hare=np.array(
					[
						30.0,
						47.2,
						70.2,
						77.4,
						36.3,
						20.6,
						18.1,
						21.4,
						22.0,
						25.4,
						27.1,
						40.3,
						57.0,
						76.6,
						52.3,
						19.5,
						11.2,
						7.6,
						14.6,
						16.2,
						24.7,
					]
				),
			)
		)
		return observed_df

	def simulate(self, parameters: dict) -> np.ndarray | pd.DataFrame:
		"""Run the simulation.

		Args:
			parameters (dict): The simulation parameters.

		Returns:
		    np.ndarray | pd.DataFrame: The simulated data.
		"""

		def dX_dt(_: np.ndarray, X: np.ndarray) -> np.ndarray:
			x, y = X
			dx_dt = parameters["alpha"] * x - parameters["beta"] * x * y
			dy_dt = -parameters["gamma"] * y + parameters["delta"] * x * y
			return np.array([dx_dt, dy_dt])

		X0 = [parameters["h0"], parameters["l0"]]
		t = (parameters["t"].min(), parameters["t"].max())
		x_y = solve_ivp(fun=dX_dt, y0=X0, t_span=t, t_eval=parameters["t"].values).y

		df = pd.DataFrame(dict(lynx=x_y[1, :], hare=x_y[0, :]))
		return df

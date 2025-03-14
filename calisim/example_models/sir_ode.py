"""Contains an implementation of an SIR epidemiological model.

An implementation of an SIR epidemiological example model.

"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from ..base import ExampleModelBase

# Refer to: https://scipython.com/book2/chapter-8-scipy/additional-examples/the-sir-epidemic-model/


class SirOdeModel(ExampleModelBase):
	"""SIR ODE model."""

	def __init__(self) -> None:
		"""SirOdeModel constructor."""
		super().__init__()

		self.GROUND_TRUTH = dict(
			beta=0.4,
			gamma=0.1,
			N=1000,
			I0=1.0,
			R0=0,
			S0=1000 - 1.0 - 0,
		)

	def get_observed_data(self) -> np.ndarray | pd.DataFrame:
		"""Retrieve observed data.

		Returns:
		    np.ndarray | pd.DataFrame: The observed data.
		"""
		observed_df = pd.DataFrame(
			[
				{"dotS": 999.000000, "dotI": 1.000000, "dotR": 0.000000},
				{"dotS": 998.534208, "dotI": 1.349201, "dotR": 0.116592},
				{"dotS": 997.906105, "dotI": 1.819995, "dotR": 0.273899},
				{"dotS": 997.059813, "dotI": 2.454180, "dotR": 0.486007},
				{"dotS": 995.919926, "dotI": 3.308098, "dotR": 0.771976},
				{"dotS": 994.385263, "dotI": 4.457212, "dotR": 1.157524},
				{"dotS": 992.323001, "dotI": 6.000336, "dotR": 1.676664},
				{"dotS": 989.556810, "dotI": 8.068764, "dotR": 2.374426},
				{"dotS": 985.847074, "dotI": 10.839686, "dotR": 3.313240},
				{"dotS": 980.887285, "dotI": 14.538527, "dotR": 4.574189},
				{"dotS": 974.297343, "dotI": 19.441000, "dotR": 6.261657},
				{"dotS": 965.559316, "dotI": 25.926958, "dotR": 8.513726},
				{"dotS": 954.030379, "dotI": 34.456348, "dotR": 11.513273},
				{"dotS": 938.959860, "dotI": 45.548099, "dotR": 15.492041},
				{"dotS": 919.505731, "dotI": 59.765558, "dotR": 20.728711},
				{"dotS": 894.809566, "dotI": 77.638272, "dotR": 27.552163},
				{"dotS": 863.682018, "dotI": 99.916771, "dotR": 36.401211},
				{"dotS": 825.323696, "dotI": 126.950323, "dotR": 47.725981},
				{"dotS": 779.485563, "dotI": 158.548020, "dotR": 61.966417},
				{"dotS": 726.468705, "dotI": 193.979006, "dotR": 79.552289},
				{"dotS": 666.883593, "dotI": 232.244338, "dotR": 100.872070},
				{"dotS": 602.857758, "dotI": 271.089717, "dotR": 126.052525},
				{"dotS": 537.098939, "dotI": 307.909136, "dotR": 154.991925},
				{"dotS": 472.026327, "dotI": 340.563156, "dotR": 187.410517},
				{"dotS": 409.770571, "dotI": 367.378908, "dotR": 222.850521},
				{"dotS": 352.173770, "dotI": 387.150095, "dotR": 260.676135},
				{"dotS": 300.766903, "dotI": 399.153311, "dotR": 300.079786},
				{"dotS": 256.098795, "dotI": 403.623358, "dotR": 340.277847},
				{"dotS": 217.957628, "dotI": 401.460171, "dotR": 380.582201},
				{"dotS": 185.880922, "dotI": 393.735050, "dotR": 420.384028},
			]
		)

		observed_df["day"] = np.arange(0, len(observed_df), 1)
		return observed_df

	def simulate(self, parameters: dict) -> np.ndarray | pd.DataFrame:
		"""Run the simulation.

		Args:
			parameters (dict): The simulation parameters.

		Returns:
		    np.ndarray | pd.DataFrame: The simulated data.
		"""

		def dX_dt(_: np.ndarray, X: np.ndarray) -> np.ndarray:
			S, I, _ = X  # noqa: E741
			dotS = -parameters["beta"] * S * I / parameters["N"]
			dotI = (
				parameters["beta"] * S * I / parameters["N"] - parameters["gamma"] * I
			)
			dotR = parameters["gamma"] * I
			return np.array([dotS, dotI, dotR])

		X0 = [parameters["S0"], parameters["I0"], parameters["R0"]]
		t = (parameters["t"].min(), parameters["t"].max())
		x_y = solve_ivp(
			fun=dX_dt, y0=X0, t_span=t, t_eval=parameters["t"].values.flatten()
		).y

		df = pd.DataFrame(dict(dotS=x_y[0, :], dotI=x_y[1, :], dotR=x_y[2, :]))
		return df

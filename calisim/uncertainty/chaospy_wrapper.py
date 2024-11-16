"""Contains the implementations for uncertainty analysis methods using
Chaospy

Implements the supported uncertainty analysis methods using
the Chaospy library.

"""

import os.path as osp

import chaospy
import gstools
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class ChaospyUncertaintyAnalysis(CalibrationWorkflowBase):
	"""The Chaospy uncertainty analysis method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			dist_instance = getattr(chaospy, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters.append(parameter)

		self.parameters = chaospy.J(*parameters)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		solvers = ["linear", "gp", "quadrature"]
		solver_name = self.specification.solver
		if solver_name not in solvers:
			raise ValueError(
				f"Unsupported Chaospy solver: {solver_name}.",
				f"Supported Chaospy solvers are {', '.join(solvers)}",
			)

		order = self.specification.order
		rule = self.specification.method

		if solver_name == "quadrature":
			nodes, weights = chaospy.generate_quadrature(
				order, self.parameters, rule=rule
			)
			X = nodes.T
		else:
			n_samples = self.specification.n_samples
			X = self.parameters.sample(n_samples, rule=rule).T

		def uncertainty_func(
			X: np.ndarray,
			observed_data: pd.DataFrame | np.ndarray,
			parameter_names: list[str],
			data_types: list[ParameterDataType],
			uncertainty_kwargs: dict,
		) -> np.ndarray:
			parameters = []
			for theta in X:
				parameter_set = {}
				for i, parameter_value in enumerate(theta):
					parameter_name = parameter_names[i]
					data_type = data_types[i]
					if data_type == ParameterDataType.CONTINUOUS:
						parameter_set[parameter_name] = parameter_value
					else:
						parameter_set[parameter_name] = int(parameter_value)
				parameters.append(parameter_set)

			simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]

			if self.specification.batched:
				results = self.call_calibration_func(
					parameters, simulation_ids, observed_data, **uncertainty_kwargs
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.call_calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**uncertainty_kwargs,
					)
					results.append(result)

			results = np.array(results)
			return results

		uncertainty_kwargs = self.get_calibration_func_kwargs()

		Y = uncertainty_func(
			X,
			self.specification.observed_data,
			self.names,
			self.data_types,
			uncertainty_kwargs,
		)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		if solver_name == "gp":
			method_kwargs["normed"] = True

		expansion = chaospy.generate_expansion(order, self.parameters, **method_kwargs)

		linear_kwargs = {"fit_intercept": False}
		linear_models = dict(
			least_squares=lm.LinearRegression(**linear_kwargs),
			elastic=lm.MultiTaskElasticNet(alpha=0.5, **linear_kwargs),
			lasso=lm.MultiTaskLasso(**linear_kwargs),
			lasso_lars=lm.LassoLars(**linear_kwargs),
			lars=lm.Lars(**linear_kwargs),
			ridge=lm.Ridge(**linear_kwargs),
		)
		linear_regression = self.specification.algorithm
		model = linear_models.get(linear_regression, None)

		if solver_name == "quadrature":
			model_approx = chaospy.fit_quadrature(
				expansion,
				nodes,  # type: ignore[possibly-undefined]
				weights,  # type: ignore[possibly-undefined]
				Y,
			)
		else:
			model_approx = chaospy.fit_regression(
				expansion,
				X.T,
				Y,
				model=model,
				retall=False,
			)

		if solver_name == "gp":
			if self.specification.flatten_Y and len(Y.shape) > 1:
				design_list = []
				for i in range(X.shape[0]):
					for j in range(Y.shape[1]):
						row = np.append(X[i], j)
						design_list.append(row)
				X = np.array(design_list)
				Y = Y.flatten()

			gp = gstools.Gaussian(dim=X.shape[-1])
			self.krige = gstools.krige.Universal(gp, X.T, Y, list(expansion))
			self.krige(X)

		self.emulator = model_approx

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()
		solver_name = self.specification.solver

		expected = chaospy.E(self.emulator, self.parameters)
		std = chaospy.Std(self.emulator, self.parameters)

		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		observed_data = self.specification.observed_data
		X = np.arange(0, expected.shape[0], 1)
		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		axes[1].plot(X, expected)
		axes[1].set_title(f"Emulated {output_label} for {solver_name} solver")
		axes[1].fill_between(X, expected - std, expected + std, alpha=0.5)

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_emulated.png")
			fig.savefig(outfile)
		else:
			fig.show()

		if solver_name == "gp":
			mu, sigma = self.krige.field, np.sqrt(self.krige.krige_var)

			obs_shape = observed_data.shape[0]
			mu_shape = mu.shape[0]
			if obs_shape != mu_shape:
				mu = mu.reshape(obs_shape, int(mu_shape / obs_shape)).mean(axis=1)
				sigma = sigma.reshape(obs_shape, int(mu_shape / obs_shape)).mean(axis=1)

			fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)
			axes[0].plot(X, observed_data)
			axes[0].set_title(f"Observed {output_label}")
			axes[1].plot(X, mu)
			axes[1].set_title(f"Emulated {output_label} for Polynomial Kriging")
			axes[1].fill_between(X, mu - sigma, mu + sigma, alpha=0.5)

			fig.tight_layout()
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{task}_polynomial_kriging.png")
				fig.savefig(outfile)
			else:
				fig.show()

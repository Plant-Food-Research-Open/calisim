"""Contains the implementations for history matching methods using
iterative_ensemble_smoother

Implements the supported history matching methods using
the iterative_ensemble_smoother library.

"""

import iterative_ensemble_smoother as ies
import numpy as np
import pandas as pd
from iterative_ensemble_smoother.utils import steplength_exponential
from matplotlib import pyplot as plt

from ..base import HistoryMatchingBase
from ..data_model import ParameterEstimateModel


class IESHistoryMatching(HistoryMatchingBase):
	"""The iterative_ensemble_smoother history matching method class."""

	def convert_parameters(self, X: np.ndarray) -> list[dict[str, float]]:
		"""Convert the parameters from an array to a list of records.

		Args:
		    X (np.ndarray): The array of parameters.

		Returns:
		    List[Dict[str, float]]: The list of parameters.
		"""
		parameters = []
		ensemble_size = self.specification.n_samples
		parameter_spec = self.specification.parameter_spec.parameters
		for i in range(ensemble_size):
			parameter_set = {}
			for j, spec in enumerate(parameter_spec):  # type: ignore[arg-type]
				parameter_name = spec.name
				parameter_set[parameter_name] = X[j][i]
			parameters.append(parameter_set)
		return parameters

	def run_simulation(self, parameters: list[dict[str, float]]) -> np.ndarray:
		"""Run the simulation for the history matching procedure.

		Args:
		    parameters (List[Dict[str, float]]): The list of
				simulation parameters.

		Returns:
		    np.ndarray: The ensemble outputs.
		"""
		observed_data = self.specification.observed_data
		history_matching_kwargs = self.get_calibration_func_kwargs()

		simulation_ids = [self.get_simulation_uuid() for _ in range(len(parameters))]

		if self.specification.batched:
			ensemble_outputs = self.call_calibration_func(
				parameters, simulation_ids, observed_data, **history_matching_kwargs
			)
		else:
			ensemble_outputs = []
			for i, parameter in enumerate(parameters):
				simulation_id = simulation_ids[i]
				outputs = self.call_calibration_func(
					parameter, simulation_id, observed_data, **history_matching_kwargs
				)
				ensemble_outputs.append(outputs)  # type: ignore[arg-type]

		ensemble_outputs = np.array(ensemble_outputs).T
		return ensemble_outputs

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		parameters = self.specification.X
		if parameters is None:
			parameters = []
			ensemble_size = self.specification.n_samples
			for i in range(ensemble_size):
				parameter_set = {}
				for k in self.parameters:
					parameter_set[k] = self.parameters[k][i]
				parameters.append(parameter_set)

		ensemble_outputs = self.specification.Y
		if ensemble_outputs is None:
			ensemble_outputs = self.run_simulation(parameters)

		smoother_name = self.specification.method
		smoothers = dict(sies=ies.SIES, esmda=ies.ESMDA)
		smoother_class = smoothers.get(smoother_name, None)
		if smoother_class is None:
			raise ValueError(
				f"Unsupported iterative ensemble smoother: {smoother_name}.",
				f"Supported iterative ensemble smoothers are {', '.join(smoothers)}",
			)

		param_values = np.array([distr for distr in self.parameters.values()])
		X_i = param_values.copy()

		n_iterations = self.specification.n_iterations
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}
		if smoother_name == "sies":
			method_kwargs["parameters"] = X_i
		else:
			method_kwargs["alpha"] = n_iterations

		observations = self.specification.observed_data
		covariance = self.specification.covariance
		if covariance is None:
			covariance = np.eye(observations.shape[0])

		smoother = smoother_class(
			covariance=covariance,
			observations=observations,
			seed=self.specification.random_seed,
			**method_kwargs,
		)

		Y_i = ensemble_outputs.copy()
		if smoother_name == "sies":
			for i, alpha_i in enumerate(range(n_iterations)):
				if self.specification.verbose:
					print(f"SIES iteration {i + 1}/{n_iterations}")
				step_length = steplength_exponential(i + 1)
				X_i = smoother.sies_iteration(Y_i, step_length=step_length)
				parameters = self.convert_parameters(X_i)
				Y_i = self.run_simulation(parameters)
		else:
			for i, alpha_i in enumerate(smoother.alpha):
				if self.specification.verbose:
					print(
						f"ESMDA iteration {i + 1}/{smoother.num_assimilations()}"
						+ f" with inflation factor alpha_i={alpha_i}"
					)

				X_i = smoother.assimilate(X_i, Y=Y_i)
				parameters = self.convert_parameters(X_i)
				Y_i = self.run_simulation(parameters)

		self.X_IES = X_i
		self.Y_IES = Y_i

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		smoother_name = self.specification.method
		n_iterations = self.specification.n_iterations

		parameter_names = list(self.parameters.keys())
		fig, axes = plt.subplots(
			nrows=len(parameter_names), figsize=self.specification.figsize
		)
		for i, parameter_name in enumerate(parameter_names):
			axes[i].set_title(parameter_name)
			axes[i].hist(self.parameters[parameter_name], label="Prior")
			axes[i].hist(
				self.X_IES[i, :],
				label=f"{smoother_name} ({n_iterations}) Posterior",
				alpha=0.5,
			)
			axes[i].legend()
		self.present_fig(fig, outdir, time_now, task, experiment_name, "plot_slice")

		ensemble_size = self.specification.n_samples
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		X = np.arange(0, self.Y_IES.shape[0], 1)
		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)

		observed_data = self.specification.observed_data
		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		for i in range(ensemble_size):
			axes[1].plot(X, self.Y_IES.T[i])
		axes[1].set_title(f"Ensemble {output_label}")
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, f"ensemble_{output_label}"
		)

		if outdir is None:
			return

		X_IES_df = pd.DataFrame(self.X_IES.T, columns=parameter_names)
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_posterior.csv"
		)
		self.append_artifact(outfile)
		X_IES_df.to_csv(outfile, index=False)

		Y_IES_df = pd.DataFrame(
			self.Y_IES.T,
			columns=[f"{output_label}_{i + 1}" for i in range(self.Y_IES.shape[0])],
		)
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_ensemble_{output_label}.csv"
		)
		self.append_artifact(outfile)
		Y_IES_df.to_csv(outfile, index=False)

		for name in X_IES_df:
			estimate = X_IES_df[name].mean()
			uncertainty = X_IES_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

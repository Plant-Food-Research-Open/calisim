"""Contains the implementations for history matching methods using
iterative_ensemble_smoother

Implements the supported history matching methods using
the iterative_ensemble_smoother library.

"""

import os.path as osp

import iterative_ensemble_smoother as ies
import numpy as np
import pandas as pd
from iterative_ensemble_smoother.utils import steplength_exponential
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..utils import get_simulation_uuid


class IESHistoryMatching(CalibrationWorkflowBase):
	"""The iterative_ensemble_smoother history matching method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		ensemble_size = self.specification.n_samples
		parameter_spec = self.specification.parameter_spec.parameters
		self.rng = np.random.default_rng(self.specification.random_seed)

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
		history_matching_kwargs = self.specification.calibration_func_kwargs
		if history_matching_kwargs is None:
			history_matching_kwargs = {}

		simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]

		if self.specification.vectorize:
			ensemble_outputs = self.calibration_func(
				parameters, simulation_ids, observed_data, **history_matching_kwargs
			)
		else:
			ensemble_outputs = []
			for i, parameter in enumerate(parameters):
				simulation_id = simulation_ids[i]
				outputs = self.calibration_func(
					parameter, simulation_id, observed_data, **history_matching_kwargs
				)
				ensemble_outputs.append(outputs)

		ensemble_outputs = np.array(ensemble_outputs).T
		return ensemble_outputs

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		parameters = []
		ensemble_size = self.specification.n_samples
		for i in range(ensemble_size):
			parameter_set = {}
			for k in self.parameters:
				parameter_set[k] = self.parameters[k][i]
			parameters.append(parameter_set)
		ensemble_outputs = self.run_simulation(parameters)

		smoother_name = self.specification.method
		smoothers = dict(sies=ies.SIES, esmda=ies.ESMDA)
		smoother_class = smoothers.get(smoother_name, None)
		if smoother_class is None:
			raise ValueError(
				f"Unsupported iterative ensemble smoother: {smoother_name}"
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
		task, time_now, outdir = self.prepare_analyze()
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

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_plot_slice.png")
			fig.savefig(outfile)
		else:
			fig.show()

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

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_ensemble_{output_label}.png")
			fig.savefig(outfile)
		else:
			fig.show()

		if outdir is None:
			return

		X_IES_df = pd.DataFrame(self.X_IES.T, columns=parameter_names)
		outfile = osp.join(outdir, f"{time_now}_{task}_posterior.csv")
		X_IES_df.to_csv(outfile, index=False)

		Y_IES_df = pd.DataFrame(
			self.Y_IES.T,
			columns=[f"{output_label}_{i + 1}" for i in range(self.Y_IES.shape[0])],
		)
		outfile = osp.join(outdir, f"{time_now}_{task}_ensemble_{output_label}.csv")
		Y_IES_df.to_csv(outfile, index=False)

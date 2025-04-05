"""Contains the implementations for state estimation using
Ensemble Kalman Filter

Implements the supported state estimation using
the Ensemble Kalman Filter.

"""

# mypy: disable-error-code="index, arg-type"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ...base import HistoryMatchingBase

# See https://github.com/ajwdewit/pcse_notebooks/blob/master/08a%20Data%20assimilation%20with%20the%20EnKF.ipynb


class EKFStateEstimation(HistoryMatchingBase):
	"""The Ensemble Kalman Filter state estimation method class."""

	def get_ensemble(self) -> dict[str, dict]:
		"""Return the ensemble.

		Returns:
		    dict[str, list]: The ensemble.
		"""
		return self.ensemble

	def initialise_ensemble(self) -> None:
		"""Initialise the state of all ensemble members."""
		ensemble_size = self.specification.n_samples
		observed_data = self.specification.observed_data
		outputs = self.specification.output_labels
		state_estimation_kwargs = self.get_calibration_func_kwargs()

		self.t = -1
		self.observed_indx = 0
		self.perform_update = False
		self.ensemble = {}

		for i in range(ensemble_size):
			simulation_id = self.get_simulation_uuid()
			parameters: dict[str, float] = {}
			self.ensemble[simulation_id] = dict(
				parameters=parameters, model=None, result={}
			)

			for output in outputs:
				self.ensemble[simulation_id]["result"][output] = []  # type: ignore[assignment]

			for k in self.parameters:
				parameters[k] = self.parameters[k][i]

			self.call_calibration_func(
				parameters, simulation_id, observed_data, **state_estimation_kwargs
			)

		self.t = 0

	def update_step(self) -> None:
		"""Perform the update/analyse step."""
		outputs = self.specification.output_labels
		ensemble_size = self.specification.n_samples
		observed_data = self.specification.observed_data
		stds = self.specification.stds

		collected_states = []
		for simulation_id in self.ensemble:
			state = {}
			for output in outputs:
				state_variable = self.ensemble[simulation_id]["result"][output][-1]
				state[output] = state_variable
			collected_states.append(state)

		df_A = pd.DataFrame(collected_states)
		A = np.matrix(df_A).T
		if np.isnan(A).any():
			return
		P_e = np.matrix(df_A.cov())

		perturbed_observations = []
		for output in outputs:
			observed_value = observed_data[output].iloc[self.observed_indx]
			std = abs(observed_value * stds[output])
			sample = self.rng.normal(observed_value, std, (ensemble_size))
			perturbed_observations.append(sample)

		df_perturbed_observations = pd.DataFrame(perturbed_observations).T
		df_perturbed_observations.columns = outputs
		D = np.matrix(df_perturbed_observations).T
		R_e = np.matrix(df_perturbed_observations.cov())

		H = np.identity(len(outputs))
		K1 = P_e * (H.T)
		K2 = (H * P_e) * H.T
		K = K1 * (np.linalg.pinv(K2 + R_e))
		Aa = A + K * (D - (H * A))
		df_Aa = pd.DataFrame(Aa.T, columns=outputs)

		for i, simulation_id in enumerate(self.ensemble):
			for output in outputs:
				self.ensemble[simulation_id]["result"][output][-1] = df_Aa[output].iloc[
					i
				]

		self.perform_update = False
		self.observed_indx += 1

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		pass_calibration_workflow = self.specification.pass_calibration_workflow
		if pass_calibration_workflow is None:
			self.specification.pass_calibration_workflow = True

		observed_data = self.specification.observed_data
		state_estimation_kwargs = self.get_calibration_func_kwargs()

		self.initialise_ensemble()

		n_iterations = self.specification.n_iterations
		for _ in range(n_iterations):
			for simulation_id in self.ensemble:
				parameters = self.ensemble[simulation_id]["parameters"]
				self.call_calibration_func(
					parameters, simulation_id, observed_data, **state_estimation_kwargs
				)

			if self.perform_update:
				self.update_step()

			self.t += 1

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		outputs = self.specification.output_labels

		results: dict[str, list] = dict(simulation_id=[], index=[])
		for output in outputs:
			results[output] = []

		for simulation_id in self.ensemble:
			result: list[float] | float = []
			for output in outputs:
				result = self.ensemble[simulation_id]["result"][output]
				results[output].extend(result)

			result_len = len(result)
			simulation_ids = [simulation_id for _ in range(result_len)]
			results["simulation_id"].extend(simulation_ids)
			results["index"].extend(np.arange(0, result_len, 1).tolist())
		results_df = pd.DataFrame(results)

		for output in outputs:
			fig, axes = plt.subplots(nrows=1, figsize=self.specification.figsize)

			for _, df in results_df.groupby("simulation_id"):
				df.plot.scatter(x="index", y=output, ax=axes)

			self.present_fig(
				fig, outdir, time_now, task, experiment_name, f"ensemble-{output}"
			)

		def q1(x: float) -> float:
			return x.quantile(0.025)

		def q2(x: float) -> float:
			return x.quantile(0.5)

		def q3(x: float) -> float:
			return x.quantile(0.975)

		quantile_list = [q1, q2, q3]
		for output in outputs:
			agg = {output: quantile_list}
			quantile_df = (
				results_df.groupby("index", as_index=False)
				.agg(agg)
				.droplevel(axis=1, level=0)
				.reset_index()
			)

			fig, axes = plt.subplots(nrows=1, figsize=self.specification.figsize)
			for quantile in quantile_list:
				quantile_df[quantile.__name__].plot(ax=axes)

			self.present_fig(
				fig,
				outdir,
				time_now,
				task,
				experiment_name,
				f"ensemble-quantile-{output}",
			)

		if outdir is None:
			return

		results_df = results_df.drop(columns="index")
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_ensemble.csv")
		self.append_artifact(outfile)
		results_df.to_csv(outfile, index=False)

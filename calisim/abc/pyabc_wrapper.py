"""Contains the implementations for Approximate Bayesian Computation methods using
PyABC

Implements the supported Approximate Bayesian Computation methods using
the PyABC library.

"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pyabc
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType, ParameterEstimateModel


class PyABCApproximateBayesianComputation(CalibrationWorkflowBase):
	"""The PyABC Approximate Bayesian Computation method class."""

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
		distributions = {}
		transition_mapping = {}
		parameter_spec = self.specification.parameter_spec.parameters

		for spec in parameter_spec:
			parameter_name = spec.name
			data_type = spec.data_type
			if data_type == ParameterDataType.DISCRETE:
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				lower_bound = np.floor(lower_bound).astype("int")
				upper_bound = np.floor(upper_bound).astype("int")
				discrete_domain = np.arange(lower_bound, upper_bound + 1)

				distributions[parameter_name] = pyabc.RV(
					"rv_discrete",
					values=(
						discrete_domain,
						np.repeat(1 / len(discrete_domain), len(discrete_domain)),
					),
				)
				transition_mapping[parameter_name] = pyabc.DiscreteJumpTransition(
					domain=discrete_domain, p_stay=0.7
				)
			else:
				distribution_name = self.dist_name_processing(spec.distribution_name)
				distribution_args = spec.distribution_args
				if distribution_args is None:
					distribution_args = []

				distribution_kwargs = spec.distribution_kwargs
				if distribution_kwargs is None:
					distribution_kwargs = {}

				distributions[parameter_name] = pyabc.RV(
					distribution_name, *distribution_args, **distribution_kwargs
				)
				transition_mapping[parameter_name] = pyabc.MultivariateNormalTransition(
					scaling=1
				)

		self.parameters = pyabc.Distribution(**distributions)
		self.transitions = pyabc.AggregatedTransition(mapping=transition_mapping)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		adaptive_pop_size = pyabc.AdaptivePopulationSize(
			self.specification.n_init,
			self.specification.min_population_size,
			self.specification.n_bootstrap,
		)

		output_labels = self.specification.output_labels
		abc_kwargs = self.get_calibration_func_kwargs()

		def simulator_func(parameters: dict) -> dict:
			observed_data = self.specification.observed_data
			simulation_id = self.get_simulation_uuid()

			results = self.call_calibration_func(
				parameters, simulation_id, observed_data, **abc_kwargs
			)

			summary_stats = {}
			if len(output_labels) == 1:  # type: ignore[arg-type]
				summary_stats[output_labels[0]] = results  # type: ignore[index]
			else:
				for i, output_label in enumerate(output_labels):  # type: ignore[arg-type]
					summary_stats[output_label] = results[i]  # type: ignore[index]

			return summary_stats

		if self.specification.n_jobs > 1:
			sampler = pyabc.MulticoreEvalParallelSampler
		else:
			sampler = pyabc.SingleCoreSampler(check_max_eval=True)

		distance_func = pyabc.AggregatedDistance(
			[
				lambda simulated, _: simulated[output_label]
				for output_label in output_labels
			]
		)

		self.abc = pyabc.ABCSMC(
			simulator_func,
			self.parameters,
			distance_func,
			population_size=adaptive_pop_size,
			transitions=self.transitions,
			eps=pyabc.MedianEpsilon(),
			sampler=sampler,
		)

		self.abc.new("sqlite://")

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		self.history = self.abc.run(
			minimum_epsilon=self.specification.epsilon,
			max_walltime=timedelta(minutes=self.specification.walltime),
			**method_kwargs,
		)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		for plot_func in [
			pyabc.visualization.plot_sample_numbers,
			pyabc.visualization.plot_total_sample_numbers,
			pyabc.visualization.plot_sample_numbers_trajectory,
			pyabc.visualization.plot_epsilons,
			pyabc.visualization.plot_effective_sample_sizes,
			pyabc.visualization.plot_walltime,
			pyabc.visualization.plot_total_walltime,
			pyabc.visualization.plot_contour_matrix,
			pyabc.visualization.plot_acceptance_rates_trajectory,
			pyabc.visualization.plot_kde_matrix_highlevel,
		]:
			abc_plot = plot_func(self.history)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-{plot_func.__name__}.png",
				)
				self.append_artifact(outfile)
				plt.tight_layout()
				plt.savefig(outfile)
				plt.close()
			else:
				abc_plot.show()

		if outdir is None:
			return

		distribution_dfs = []
		for t in range(self.history.max_t + 1):
			df, w = self.history.get_distribution(m=0, t=t)
			df["t"] = t
			df["w"] = w
			distribution_dfs.append(df)

		distribution_dfs = pd.concat(distribution_dfs)
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_parameters.csv"
		)
		self.append_artifact(outfile)
		distribution_dfs.to_csv(outfile, index=False)

		populations_df = self.history.get_all_populations()
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_populations.csv"
		)
		self.append_artifact(outfile)
		populations_df.to_csv(outfile, index=False)

		population_particles_df = self.history.get_nr_particles_per_population()
		outfile = self.join(
			outdir,
			f"{time_now}-{task}-{experiment_name}_nr_particles_per_population.csv",
		)
		self.append_artifact(outfile)
		population_particles_df.to_csv(outfile, index=False)

		distances_df = []
		for t in range(self.history.max_t + 1):
			df = self.history.get_weighted_distances(t=t)
			df["t"] = t
			distances_df.append(df)
		distances_df = pd.concat(distances_df)

		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_distances.csv"
		)
		self.append_artifact(outfile)
		distances_df.to_csv(outfile, index=False)

		max_t = distribution_dfs["t"].max()
		trace_df = distribution_dfs.query(f"t == {max_t}")
		trace_df = trace_df.drop(columns=["t", "w"])

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

"""Contains the implementations for Approximate Bayesian Computation methods using
PyMC

Implements the supported Approximate Bayesian Computation methods using
the PyMC library.

"""

from collections.abc import Callable

import arviz as az
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterEstimateModel


class PyMCApproximateBayesianComputation(CalibrationWorkflowBase):
	"""The PyMC Approximate Bayesian Computation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		priors = []
		parameter_spec = self.specification.parameter_spec.parameters
		with pm.Model() as self.model:
			for spec in parameter_spec:
				parameter_name = spec.name
				self.names.append(parameter_name)

				distribution_name = (
					spec.distribution_name.replace("_", " ").title().replace(" ", "")
				)

				distribution_class = getattr(pm, distribution_name)
				distribution_args = spec.distribution_args
				if distribution_args is None:
					distribution_args = []

				distribution_kwargs = spec.distribution_kwargs
				if distribution_kwargs is None:
					distribution_kwargs = {}

				prior = distribution_class(
					parameter_name, *distribution_args, **distribution_kwargs
				)
				priors.append(prior)
			self.parameters = tuple(priors)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		distance = self.specification.distance
		if distance is None:
			distance = lambda e, _, simulated: simulated.item()  # noqa: E731

		def simulator_func(
			_: np.random.Generator, *parameter_values: list
		) -> np.ndarray:
			parameters = {}
			for i, name in enumerate(self.names):
				parameters[name] = parameter_values[i].item()

			abc_kwargs = self.get_calibration_func_kwargs()
			simulation_id = self.get_simulation_uuid()
			observed_data = self.specification.observed_data
			results = self.call_calibration_func(
				parameters, simulation_id, observed_data, **abc_kwargs
			)

			if not hasattr(results, "__iter__"):
				results = [results]
			if not isinstance(results, np.ndarray):
				results = np.array(results)

			return results

		with self.model:
			pm.Simulator(
				self.specification.experiment_name,
				simulator_func,
				params=self.parameters,
				distance=distance,
				sum_stat=self.specification.sum_stat,
				epsilon=self.specification.epsilon,
				observed=self.specification.observed_data,
			)

			method_kwargs = self.specification.method_kwargs
			if method_kwargs is None:
				method_kwargs = {}

			self.trace = pm.sample_smc(
				model=self.model,
				draws=self.specification.n_samples,
				chains=self.specification.n_chains,
				cores=self.specification.n_jobs,
				random_seed=self.specification.random_seed,
				**method_kwargs,
			)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		textsize = 7
		for plot in ["trace", "rank_vlines", "rank_bars"]:
			fig = az.plot_trace(
				self.trace, kind=plot, plot_kwargs={"textsize": textsize}
			)

			if outdir is not None:
				outfile = self.join(
					outdir, f"{time_now}-{task}-{experiment_name}_{plot}.png"
				)
				self.append_artifact(outfile)
				plt.tight_layout()
				plt.savefig(outfile)
				plt.close()
			else:
				fig.show()

		def _create_plot(plot_func: Callable, plot_kwargs: dict) -> None:
			plot_func(self.trace, **plot_kwargs)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}_{plot_func.__name__}.png",
				)
				self.append_artifact(outfile)
				plt.tight_layout()
				plt.savefig(outfile)
				plt.close()
			else:
				fig.show()

		_create_plot(
			az.plot_pair,
			plot_kwargs={
				"figsize": (12, 12),
				"scatter_kwargs": dict(alpha=0.01),
				"marginals": True,
				"textsize": textsize,
			},
		)

		_create_plot(
			az.plot_violin, plot_kwargs={"figsize": (12, 12), "textsize": textsize}
		)

		_create_plot(
			az.plot_posterior, plot_kwargs={"figsize": (12, 12), "textsize": 5}
		)

		if outdir is None:
			return

		trace_df = self.trace.to_dataframe(include_coords=False, groups="posterior")
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trace.csv")
		self.append_artifact(outfile)
		trace_df.to_csv(outfile, index=False)

		trace_summary_df = az.summary(self.trace).reset_index()
		trace_summary_df = trace_summary_df.rename(columns={"index": "name"})
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_trace_summary.csv"
		)
		self.append_artifact(outfile)
		trace_summary_df.to_csv(outfile, index=False)

		for row in trace_summary_df.to_dict("records"):
			name = row["name"]
			estimate = row["mean"]
			uncertainty = row["sd"]
			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

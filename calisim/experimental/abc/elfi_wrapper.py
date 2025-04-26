"""Contains the implementations for Approximate Bayesian Computation methods using
ELFI

Implements the supported Approximate Bayesian Computation methods using
the ELFI library.

"""

import time

import elfi
import numpy as np
from matplotlib import pyplot as plt

from ...data_model import ParameterEstimateModel
from ..base import ELFIBase


class ELFIApproximateBayesianComputation(ELFIBase):
	"""The ELFI Approximate Bayesian Computation method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		def simulator_func(
			*parameter_values: list,
			size: int | None = None,
			batch_size: int = 1,
			random_state: int | None = None,
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

		self.create_simulator(simulator_func, take_log=False)

		sampler_name = self.specification.method
		supported_samplers = dict(
			rejection=elfi.Rejection,
			smc=elfi.SMC,
			adaptive_threshold_smc=elfi.AdaptiveThresholdSMC,
		)
		sampler_class = supported_samplers.get(sampler_name, None)
		if sampler_class is None:
			raise ValueError(
				f"Unsupported ELFI sampler: {sampler_name}.",
				f"Supported ELFI samplers are {', '.join(supported_samplers)}",
			)

		sampler_kwargs = self.specification.method_kwargs
		if sampler_kwargs is None:
			sampler_kwargs = {}

		random_seed = self.specification.random_seed
		sampler_kwargs["batch_size"] = 1
		sampler_kwargs["seed"] = random_seed
		self.sampler = sampler_class(self.distance, **sampler_kwargs)

		n_samples = self.specification.n_samples
		quantile = self.specification.quantile
		n_iterations = self.specification.n_iterations
		self.sampler.bar = True
		if sampler_name == "rejection":
			self.sampler.set_objective(n_samples, quantile=quantile)
		elif sampler_name == "smc":
			quantiles = [quantile for _ in range(n_iterations)]
			self.sampler.set_objective(n_samples, quantiles=quantiles)
		elif sampler_name == "adaptive_threshold_smc":
			self.sampler.set_objective(n_samples, max_iter=n_iterations)
		else:
			raise ValueError(f"Unsupported ELFI sampler: {sampler_name}")

		walltime = self.specification.walltime
		walltime = time.time() + walltime * 60
		while not self.sampler.finished and time.time() < walltime:
			self.sampler.iterate()

		self.history = self.sampler.extract_result()

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		n_samples = self.specification.n_samples
		num_bins = 25
		if n_samples <= 20:
			num_bins = n_samples

		for plot_func in [self.history.plot_marginals, self.history.plot_pairs]:
			plot_func(bins=num_bins, figsize=self.specification.figsize)
			if outdir is not None:
				plot_name = plot_func.__name__.replace("_", "-")
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-{plot_name}.png",
				)
				self.append_artifact(outfile)
				plt.tight_layout()
				plt.savefig(outfile)
				plt.close()
			else:
				plt.show()
				plt.close()

		for name, values in self.history.samples.items():
			estimate = values.mean()  # Unweighted samples
			uncertainty = values.std()
			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

		if outdir is None:
			return

		outfile = self.join(
			outdir,
			f"{time_now}-{task}-{experiment_name}-trace.csv",
		)
		self.append_artifact(outfile)
		self.history.save(outfile)

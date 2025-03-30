"""Contains the implementations for likelihood-free methods using
ELFI

Implements the supported likelihood-free methods using
the ELFI library.

"""

import time

import elfi
import numpy as np
from matplotlib import pyplot as plt

from ..base import ELFIBase
from ..data_model import ParameterEstimateModel


class ELFILikelihoodFree(ELFIBase):
	"""The ELFI likelihood-free method class."""

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
			bolfi=elfi.BOLFI,
		)
		sampler_class = supported_samplers.get(sampler_name, None)
		if sampler_class is None:
			raise ValueError(
				f"Unsupported ELFI sampler: {sampler_name}.",
				f"Supported ELFI samplers are {', '.join(supported_samplers)}",
			)

		n_init = self.specification.n_init
		sampler_kwargs = dict(
			target_name=self.distance,
			initial_evidence=n_init,
			batch_size=1,
			update_interval=n_init // 10,
			acq_noise_var=self.specification.acq_noise_var,
			bounds=self.bounds,
			seed=self.specification.random_seed,
		)

		self.sampler = sampler_class(self.distance.model, **sampler_kwargs)

		n_iterations = self.specification.n_iterations
		objective_kwargs = dict(n_evidence=n_iterations)
		self.sampler.set_objective(**objective_kwargs)

		walltime = self.specification.walltime
		walltime = time.time() + walltime * 60
		while not self.sampler.finished and time.time() < walltime:
			self.sampler.iterate()

		n_samples = self.specification.n_samples
		n_chains = self.specification.n_chains
		sampler = self.specification.sampler
		epsilon = self.specification.epsilon

		self.history = self.sampler.sample(
			n_samples, algorithm=sampler, n_chains=n_chains, threshold=epsilon
		)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		for plot_func in [
			self.sampler.plot_state,
			self.sampler.plot_discrepancy,
			self.sampler.plot_gp,
			self.history.plot_traces,
			self.history.plot_marginals,
			self.history.plot_pairs,
		]:
			plot_func(figsize=self.specification.figsize)
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
				plt.show()

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

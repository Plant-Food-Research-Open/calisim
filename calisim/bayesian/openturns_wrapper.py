"""Contains the implementations for Bayesian calibration methods using
OpenTurns

Implements the supported Bayesian calibration methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot
import openturns.viewer as viewer
import pandas as pd

from ..base import OpenTurnsBase
from ..data_model import ParameterEstimateModel


class OpenTurnsBayesianCalibration(OpenTurnsBase):
	"""The OpenTurns Bayesian calibration method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.bounds: tuple[list[float], list[float]] = ([], [])
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			bounds = self.get_parameter_bounds(spec)
			lower_bound, upper_bound = bounds
			lower_bounds, upper_bounds = self.bounds
			lower_bounds.append(lower_bound)
			upper_bounds.append(upper_bound)

			data_type = spec.data_type
			self.data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			distribution_args = list(distribution_kwargs.values())
			dist_instance = getattr(ot, distribution_name)
			parameter = dist_instance(*distribution_args)
			parameters.append(parameter)

		distribution_collection = ot.DistributionCollection(parameters)
		self.parameters = ot.JointDistribution(distribution_collection)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		bayesian_calibration_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				bayesian_calibration_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)
			return Y

		def log_target_function(X: np.ndarray) -> np.ndarray:
			return np.log(target_function(X))

		if self.specification.log_density:
			ot_func_wrapper = self.get_ot_func_wrapper(log_target_function)
		else:
			ot_func_wrapper = self.get_ot_func_wrapper(target_function)

		memoized_func = ot.MemoizeFunction(ot_func_wrapper)
		lower_bounds, upper_bounds = self.bounds
		support = ot.Interval(lower_bounds, upper_bounds)

		initial_state = self.specification.initial_state
		if initial_state is None:
			initial_state = []
			for i, lower_bound in enumerate(lower_bounds):
				upper_bound = upper_bounds[i]
				midpoint = np.median((lower_bound, upper_bound))
				initial_state.append(midpoint)

		self.sampler = ot.IndependentMetropolisHastings(
			memoized_func, support, initial_state, self.parameters
		)

		n_samples = self.specification.n_samples
		self.sample = self.sampler.getSample(n_samples)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		sample = np.array(self.sample)
		n_dim = self.parameters.getDimension()

		kernel = ot.KernelSmoothing()
		posterior = kernel.build(sample)
		grid = ot.GridLayout(n_dim, 1)
		grid.setTitle("Bayesian inference")

		for parameter_index in range(n_dim):
			graph = posterior.getMarginal(parameter_index).drawPDF()
			prior_graph = self.parameters.getMarginal(parameter_index).drawPDF()
			graph.add(prior_graph)
			parameter_name = self.names[parameter_index]
			graph.setTitle(parameter_name)
			graph.setLegends(["Posterior", "Prior"])
			grid.setGraph(parameter_index, 0, graph)
		view = viewer.View(grid)
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}_plot_posterior.png"
			)
			self.append_artifact(outfile)
			view.save(outfile)

		if outdir is None:
			return

		trace_df = pd.DataFrame(sample, columns=self.names)
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trace.csv")
		self.append_artifact(outfile)
		trace_df.to_csv(outfile, index=False)

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

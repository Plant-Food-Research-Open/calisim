"""Contains the implementations for reliability analysis methods using
OpenTurns

Implements the supported reliability analysis methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot
import openturns.viewer as viewer

from ..base import OpenTurnsBase


class OpenTurnsReliabilityAnalysis(OpenTurnsBase):
	"""The OpenTurns reliability analysis method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		method = self.specification.method
		supported_methods = ["monte_carlo", "sobol", "subset"]
		if method not in supported_methods:
			raise ValueError(
				f"Unsupported sampler: {method}.",
				f"Supported samplers: {', '.join(supported_methods)}",
			)

		comparison = self.specification.comparison
		valid_comparisons = [
			"less",
			"less_or_equal",
			"greater",
			"greater_or_equal",
			"equal",
		]
		if comparison not in valid_comparisons:
			raise ValueError(
				f"Unsupported comparison: {comparison}.",
				f"Supported comparisons: {', '.join(valid_comparisons)}",
			)

		reliability_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				reliability_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)
			return Y

		self.input_vector = ot.RandomVector(self.parameters)
		ot_func_wrapper = self.get_ot_func_wrapper(target_function)
		memoized_func = ot.MemoizeFunction(ot_func_wrapper)
		G = ot.CompositeRandomVector(memoized_func, self.input_vector)

		threshold = self.specification.threshold
		comparison = comparison.replace("_", " ").title().replace(" ", "")
		comparison_func = getattr(ot, comparison)

		event = ot.ThresholdEvent(G, comparison_func(), threshold)

		if method == "monte_carlo":
			self.experiment = ot.MonteCarloExperiment()
		elif method == "sobol":
			sequence = ot.SobolSequence()
			self.experiment = ot.LowDiscrepancyExperiment(sequence, 1)
			self.experiment.setRandomize(True)

		if method == "subset":
			self.sampler = ot.SubsetSampling(event)
			self.sampler.setKeepSample(True)
		else:
			self.sampler = ot.ProbabilitySimulationAlgorithm(event, self.experiment)
			self.sampler.setMaximumCoefficientOfVariation(0.05)

		n_samples = self.specification.n_samples
		n_jobs = self.specification.n_jobs

		self.sampler.setMaximumOuterSampling(n_samples)
		self.sampler.setBlockSize(n_jobs)
		self.sampler.run()

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		method = self.specification.method

		result = self.sampler.getResult()
		graph = result.drawImportanceFactors()
		view = viewer.View(graph)
		if outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-{method}_plot_importance_factors.png",
			)
			self.append_artifact(outfile)
			view.save(outfile)

		graph = self.sampler.drawProbabilityConvergence()
		graph.setLogScale(ot.GraphImplementation.LOGX)
		view = viewer.View(graph)
		if outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-{method}_plot_probability_convergence.png",
			)
			self.append_artifact(outfile)
			view.save(outfile)

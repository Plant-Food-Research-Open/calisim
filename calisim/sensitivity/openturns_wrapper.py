"""Contains the implementations for sensitivity analysis methods using
OpenTurns

Implements the supported sensitivity analysis methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot
import openturns.viewer as viewer

from ..base import OpenTurnsBase
from ..estimators import FunctionalChaosEstimator


class OpenTurnsSensitivityAnalysis(OpenTurnsBase):
	"""The OpenTurns sensitivity analysis method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		method = self.specification.method
		sobol_methods = dict(
			saltelli=ot.SaltelliSensitivityAlgorithm,
			martinez=ot.MartinezSensitivityAlgorithm,
			jansen=ot.JansenSensitivityAlgorithm,
			mauntz_kucherenko=ot.MauntzKucherenkoSensitivityAlgorithm,
		)
		self.sobol_method_names = list(sobol_methods.keys())
		self.supported_methods = self.sobol_method_names.copy()
		self.supported_methods.extend(["chaos_sobol", "chaos_ancova"])

		if method not in self.supported_methods:
			raise ValueError(
				f"Unsupported sensitivity analysis algorithm: {method}.",
				f"Supported algorithms are: {', '.join(self.supported_methods)}",
			)

		sensitivity_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				sensitivity_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)
			return Y

		n_samples = self.specification.n_samples

		X = self.specification.X
		if X is None:
			sobol_experiment = ot.SobolIndicesExperiment(self.parameters, n_samples)
			X = sobol_experiment.generate()
		Y = self.specification.Y
		if Y is None:
			ot_func_wrapper = self.get_ot_func_wrapper(target_function)
			Y = ot_func_wrapper(X)

		if method in self.sobol_method_names:
			self.sp = sobol_methods.get(method)(X, Y, n_samples)
			return

		estimator = FunctionalChaosEstimator(self.parameters, self.specification.order)

		estimator.fit(X, Y)
		self.emulator = estimator

		if method == "chaos_sobol":
			self.sp = ot.FunctionalChaosSobolIndices(self.emulator.result)
		elif method == "chaos_ancova":
			self.sp = ot.ANCOVA(self.emulator.result, X)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		method = self.specification.method
		input_dim = self.parameters.getDimension()

		if method in self.sobol_method_names:
			graph = self.sp.draw()
			graph.setXTitle(" ".join(self.names))
			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-sobol_{method}_indices.png",
				)
				self.append_artifact(outfile)
				view.save(outfile)
		elif method == "chaos_sobol":
			first_order = [self.sp.getSobolIndex(i) for i in range(input_dim)]
			total_order = [self.sp.getSobolTotalIndex(i) for i in range(input_dim)]
			graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(
				self.names, first_order, total_order
			)

			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-chaos_sobol_indices.png",
				)
				self.append_artifact(outfile)
				view.save(outfile)
		elif method == "chaos_ancova":
			indices = self.sp.getIndices()
			uncorrelated_indices = self.sp.getUncorrelatedIndices()
			correlated_indices = indices - uncorrelated_indices

			graph = ot.SobolIndicesAlgorithm.DrawImportanceFactors(
				indices, self.parameters.getDescription(), "ANCOVA indices (Sobol)"
			)
			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir, f"{time_now}-{task}-{experiment_name}-ancova_indices.png"
				)
				self.append_artifact(outfile)
				view.save(outfile)

			graph = ot.SobolIndicesAlgorithm.DrawImportanceFactors(
				uncorrelated_indices,
				self.parameters.getDescription(),
				"ANCOVA uncorrelated indices",
			)
			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-uncorrelated_ancova_indices.png",
				)
				self.append_artifact(outfile)
				view.save(outfile)

			graph = ot.SobolIndicesAlgorithm.DrawImportanceFactors(
				correlated_indices,
				self.parameters.getDescription(),
				"ANCOVA correlated indices",
			)
			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-correlated_ancova_indices.png",
				)
				self.append_artifact(outfile)
				view.save(outfile)
		else:
			raise ValueError(
				f"Unsupported sensitivity analysis algorithm: {method}.",
				f"Supported algorithms are: {', '.join(self.supported_methods)}",
			)

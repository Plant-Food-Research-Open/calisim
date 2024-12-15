"""Contains the implementations for uncertainty analysis methods using
OpenTurns

Implements the supported uncertainty analysis methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot
import openturns.viewer as viewer

from ..base import OpenTurnsBase
from ..estimators import FunctionalChaosEstimator, KrigingEstimator


class OpenTurnsUncertaintyAnalysis(OpenTurnsBase):
	"""The OpenTurns uncertainty analysis method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		uncertainty_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				uncertainty_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)
			return Y

		n_samples = self.specification.n_samples
		X, Y = self.get_X_Y(n_samples, target_function)

		test_size = self.specification.test_size
		if test_size > 0:
			test_indx = int(test_size * len(X))
			self.X_test = X[:test_indx, :]
			self.Y_test = Y[:test_indx, :]

			X = X[test_indx:, :]
			Y = Y[test_indx:, :]
		else:
			self.X_test = None
			self.Y_test = None

		solver_name = self.specification.solver
		if solver_name == "functional_chaos":
			estimator = FunctionalChaosEstimator(
				self.parameters, self.specification.order
			)
		elif solver_name == "kriging":
			method_kwargs = self.specification.method_kwargs
			if method_kwargs is None:
				method_kwargs = {}

			method_kwargs["covariance_scale"] = method_kwargs.get(
				"covariance_scale", 1.0
			)
			method_kwargs["covariance_amplitude"] = method_kwargs.get(
				"covariance_amplitude", 1.0
			)
			method_kwargs["parameters"] = self.parameters
			method_kwargs["n_out"] = self.specification.n_out
			estimator = KrigingEstimator(**method_kwargs)
		else:
			raise ValueError(
				"Solver must be 'functional_chaos' or 'kriging' \
					for OpenTurnsUncertaintyAnalysis execute()"
			)

		estimator.fit(X, Y)
		self.emulator = estimator

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		solver_name = self.specification.solver
		input_dim = self.parameters.getDimension()

		solver_name = self.specification.solver
		if solver_name == "functional_chaos":
			sobol_indices = ot.FunctionalChaosSobolIndices(self.emulator.result)

			first_order = [sobol_indices.getSobolIndex(i) for i in range(input_dim)]
			total_order = [
				sobol_indices.getSobolTotalIndex(i) for i in range(input_dim)
			]
			graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(
				self.names, first_order, total_order
			)

			view = viewer.View(graph)
			if outdir is not None:
				outfile = self.join(
					outdir, f"{time_now}-{task}-{experiment_name}-sobol_indices.png"
				)
				self.append_artifact(outfile)
				view.save(outfile)

		if self.X_test is not None and self.Y_test is not None:
			y_pred = self.emulator.predict(self.X_test)
			val = ot.MetaModelValidation(self.Y_test, y_pred)
			graph = val.drawValidation()
			view = viewer.View(graph)
			r2 = self.emulator.score(self.X_test, self.Y_test)
			graph.setTitle(f"R2: {r2}")

			if outdir is not None:
				outfile = self.join(
					outdir, f"{time_now}-{task}-{experiment_name}-r2.png"
				)
				self.append_artifact(outfile)
				view.save(outfile)

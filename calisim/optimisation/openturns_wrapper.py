"""Contains the implementations for uncertainty analysis methods using
OpenTurns

Implements the supported uncertainty analysis methods using
the OpenTurns library.

"""

import numpy as np
import openturns as ot
import openturns.viewer as viewer
import pandas as pd

from ..base import OpenTurnsBase
from ..estimators import KrigingEstimator


class OpenTurnsOptimisation(OpenTurnsBase):
	"""The OpenTurns optimisation method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		optimisation_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				optimisation_kwargs,
			)
			if len(Y.shape) == 1:
				Y = np.expand_dims(Y, axis=1)
			return Y

		ot_func_wrapper = self.get_ot_func_wrapper(target_function)
		self.problem = ot.OptimizationProblem(ot_func_wrapper)
		lower_bounds, upper_bounds = self.bounds
		bounds = ot.Interval(lower_bounds, upper_bounds)
		self.problem.setBounds(bounds)

		n_init = self.specification.n_init
		n_iterations = self.specification.n_iterations
		method = self.specification.method
		if method == "kriging":
			X, Y = self.get_X_Y(n_init, target_function)

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

			estimator.fit(X, Y)
			self.emulator = estimator

			self.study = ot.EfficientGlobalOptimization(self.problem, estimator.result)
			self.study.setMaximumCallsNumber(n_iterations)
			self.study.run()
		else:
			X = self.specification.X
			if X is None:
				X = self.sample_parameters(n_init)

			self.study = ot.Pagmo(self.problem, method, X)
			self.study.setMaximumIterationNumber(n_iterations)
			self.study.run()

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()
		method = self.specification.method

		result = self.study.getResult()
		graph = result.drawOptimalValueHistory()
		view = viewer.View(graph)
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}_plot_optimization_history.png"
			)
			view.save(outfile)

		graph = result.drawErrorHistory()
		view = viewer.View(graph)
		if outdir is not None:
			outfile = self.join(outdir, f"{time_now}-{task}_plot_error_history.png")
			view.save(outfile)

		if outdir is None:
			return

		parameters = result.getInputSample()
		objective_results = result.getOutputSample()

		trial_rows = []
		for i, parameter_set in enumerate(parameters):
			trial_row = dict(algorithm=method, number=i)

			objective_result = objective_results[i]
			for j, objective_result in enumerate(objective_result):
				trial_row[f"value_{j}"] = objective_result

			for j, name in enumerate(self.names):
				trial_row[f"param_{name}"] = parameter_set[j]

			trial_rows.append(trial_row)

		trials_df: pd.DataFrame = pd.DataFrame(trial_rows)
		outfile = self.join(outdir, f"{time_now}_{task}_trials.csv")
		trials_df.to_csv(outfile, index=False)

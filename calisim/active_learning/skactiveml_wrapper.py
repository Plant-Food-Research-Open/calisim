"""Contains the implementations for active learning methods using scikit-activeml

Implements the supported active learning methods using the scikit-activeml library.

"""

import numpy as np
import pandas as pd
from emukit.core.initial_designs import RandomDesign
from matplotlib import pyplot as plt
from skactiveml.pool import (
	ExpectedModelChangeMaximization,
	ExpectedModelVarianceReduction,
	GreedySamplingTarget,
	GreedySamplingX,
	KLDivergenceMaximization,
	RegressionTreeBasedAL,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from ..base import EmukitBase


class SkActiveMLActiveLearning(EmukitBase):
	"""The scikit-activeml active learning method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		active_learning_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			return self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				active_learning_kwargs,
			)

		n_init = self.specification.n_init
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		X, Y_true = self.get_X_Y(n_init, target_function)
		self.Y_shape = 1
		if len(Y_true.shape) > 1:
			self.Y_shape = Y_true.shape[1]
			if self.Y_shape > 1:
				X = self.extend_X(X, self.Y_shape)
				Y_true = Y_true.flatten()

		Y = np.full_like(Y_true, np.nan)
		surrogate_name = self.specification.method
		surrogates = dict(
			nick=NICKernelRegressor,
			gp=GaussianProcessRegressor,
		)
		surrogate_class = surrogates.get(surrogate_name, None)
		if surrogate_class is None:
			raise ValueError(
				f"Unsupported surrogate class: {surrogate_name}.",
				f"Supported surrogate classes are {', '.join(surrogates)}",
			)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		emulator = surrogate_class(**method_kwargs)
		if surrogate_name != "nick":
			emulator = SklearnRegressor(emulator)

		query_name = self.specification.query_strategy
		query_stategies = dict(
			greedy_sampling_x=GreedySamplingX,
			greedy_sampling_target=GreedySamplingTarget,
			regression_tree_based_al=RegressionTreeBasedAL,
			kl_divergence_maximization=KLDivergenceMaximization,
			expected_model_change_maximization=ExpectedModelChangeMaximization,
			expected_model_variance_reduction=ExpectedModelVarianceReduction,
		)
		query_class = query_stategies.get(query_name, None)
		if query_class is None:
			raise ValueError(
				f"Unsupported query strategy: {query_name}.",
				f"Supported query strategies are {', '.join(query_stategies)}",
			)
		query_strategy = query_class(random_state=self.specification.random_seed)

		n_iterations = self.specification.n_iterations
		for _ in range(n_iterations):
			query_idx = query_strategy.query(X=X, y=Y, reg=emulator, fit_reg=True)
			Y[query_idx] = Y_true[query_idx]

		emulator.fit(X, Y)
		self.emulator = emulator
		self.query_strategy = query_strategy
		self.Y_true = Y_true

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		design = RandomDesign(self.parameter_space)
		n_samples = self.specification.n_samples
		X_sample = design.get_samples(n_samples)
		if self.Y_shape > 1:
			X_sample = self.extend_X(X_sample, self.Y_shape)
		predicted = self.emulator.predict(X_sample)

		names = self.names.copy()
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		if X_sample.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(X_sample, columns=names)
		df[f"emulated-{output_label}"] = predicted

		if self.specification.use_shap and outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-param_importances.png",
			)
			self.calculate_shap_importances(
				X_sample, self.emulator, names, self.specification.test_size, outfile
			)

		fig, axes = plt.subplots(
			nrows=len(self.names), figsize=self.specification.figsize
		)
		for i, name in enumerate(self.names):
			df.plot.scatter(name, f"emulated-{output_label}", ax=axes[i])
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, f"emulated-{output_label}"
		)

		if outdir is None:
			return

		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}-emulated-{output_label}.csv"
		)
		self.append_artifact(outfile)
		df.to_csv(outfile, index=False)

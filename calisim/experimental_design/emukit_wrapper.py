"""Contains the implementations for experimental design methods using Emukit

Implements the supported experimental design methods using the Emukit library.

"""

import numpy as np
from emukit.core.initial_designs import RandomDesign
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.experimental_design.acquisitions import (
	IntegratedVarianceReduction,
	ModelVariance,
)
from matplotlib import pyplot as plt

from ..base import EmukitBase
from ..estimators import EmukitEstimator


class EmukitExperimentalDesign(EmukitBase):
	"""The Emukit experimental design method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		experimental_design_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			return self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				experimental_design_kwargs,
			)

		n_init = self.specification.n_init
		X, Y = self.get_X_Y(n_init, target_function)
		method_kwargs = self.specification.method_kwargs
		estimator = EmukitEstimator(method_kwargs)
		estimator.fit(X, Y)

		acquisition_name = self.specification.method
		acquisitions = dict(
			model_variance=ModelVariance,
			integrated_variance_reduction=IntegratedVarianceReduction,
		)
		acquisition_class = acquisitions.get(acquisition_name, None)
		if acquisition_class is None:
			raise ValueError(
				f"Unsupported emulator acquisition type: {acquisition_name}.",
				f"Supported acquisition types are {', '.join(acquisitions)}",
			)
		acquisition = acquisition_class(model=estimator.emulator)

		design_loop = ExperimentalDesignLoop(
			model=estimator.emulator,
			space=self.parameter_space,
			acquisition=acquisition,
			batch_size=1,
		)
		n_iterations = self.specification.n_iterations
		design_loop.run_loop(target_function, stopping_condition=n_iterations)

		self.emulator = estimator
		self.design_loop = design_loop

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		design = RandomDesign(self.parameter_space)
		n_samples = self.specification.n_samples
		X_sample = design.get_samples(n_samples)

		if self.specification.use_shap and outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-param_importances.png",
			)
			self.calculate_shap_importances(
				X_sample,
				self.emulator,
				self.names,
				self.specification.test_size,
				outfile,
			)

		predicted_mu, predicted_std = self.emulator.predict(X_sample, return_std=True)

		observed_data = self.specification.observed_data
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		X = np.arange(0, observed_data.shape[0], 1)
		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)

		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		mean_predicted_mu = predicted_mu.mean(axis=0)
		mean_predicted_std = predicted_std.mean(axis=0)
		axes[1].plot(X, mean_predicted_mu)

		for mult, alpha in [(1, 0.6), (2, 0.4), (3, 0.2)]:
			axes[1].fill_between(
				X,
				mean_predicted_mu - mult * mean_predicted_std,
				mean_predicted_mu + mult * mean_predicted_std,
				color="C0",
				alpha=alpha,
			)

		axes[1].set_title(f"Emulated {output_label}")

		self.present_fig(
			fig, outdir, time_now, task, experiment_name, f"emulated-{output_label}"
		)

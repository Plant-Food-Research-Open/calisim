"""Contains the implementations for Bayesian quadrature methods using Emukit

Implements the supported Bayesian quadrature methods using the Emukit library.

"""

import emukit.quadrature.kernels
import emukit.quadrature.measures
import numpy as np
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.quadrature.methods import VanillaBayesianQuadrature
from GPy.models import GPRegression
from matplotlib import pyplot as plt

from ..base import EmukitBase


class EmukitQuadrature(EmukitBase):
	"""The Emukit Bayesian quadrature method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		quadrature_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			return self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				quadrature_kwargs,
			)

		n_init = self.specification.n_init
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		X, Y = self.get_X_Y(n_init, target_function)
		gp = GPRegression(X, Y, **method_kwargs)
		emukit_rbf = RBFGPy(gp.kern)

		measure_name = self.specification.measure
		measure_func = getattr(emukit.quadrature.measures, measure_name)
		measure = measure_func.from_bounds(bounds=self.bounds)

		kernel_name = self.specification.kernel
		kernel_func = getattr(emukit.quadrature.kernels, kernel_name)
		kernel = kernel_func(emukit_rbf, measure)
		emulator = BaseGaussianProcessGPy(kern=kernel, gpy_model=gp)

		quadrature = VanillaBayesianQuadrature(base_gp=emulator, X=X, Y=Y)
		quadrature_loop = VanillaBayesianQuadratureLoop(model=quadrature)
		n_iterations = self.specification.n_iterations
		quadrature_loop.run_loop(target_function, stopping_condition=n_iterations)

		self.emulator = emulator
		self.quadrature_loop = quadrature_loop

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		integral_mean, integral_variance = self.quadrature_loop.model.integrate()
		fig, ax = plt.subplots(figsize=self.specification.figsize)
		ax.set_title("Integral density")
		self.rng = self.get_default_rng(self.specification.random_seed)
		integral_samples = self.rng.normal(
			integral_mean, integral_variance, size=self.specification.n_samples
		)
		ax.hist(
			integral_samples,
			alpha=0.5,
		)
		ax.legend()
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, "integral_density"
		)

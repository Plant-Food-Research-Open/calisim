"""Contains the implementations for optimisation methods using Emukit

Implements the supported optimisation methods using the Emukit library.

"""

import os.path as osp

import emukit.quadrature.kernels
import emukit.quadrature.measures
import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.quadrature.methods import VanillaBayesianQuadrature
from GPy.models import GPRegression
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class EmukitBayesianQuadrature(CalibrationWorkflowBase):
	"""The Emukit Bayesian quadrature method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		parameters = []
		self.names = []
		self.bounds = []
		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)
			bounds = self.get_parameter_bounds(spec)
			self.bounds.append(bounds)
			lower_bound, upper_bound = bounds
			data_type = spec.data_type

			if data_type == ParameterDataType.DISCRETE:
				discrete_domain = np.arange(lower_bound, upper_bound + 1)
				parameter = DiscreteParameter(parameter_name, discrete_domain)
			else:
				parameter = ContinuousParameter(
					parameter_name, lower_bound, upper_bound
				)
			parameters.append(parameter)

		self.parameter_space = ParameterSpace(parameters)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		bayesian_quadrature_kwargs = self.specification.calibration_func_kwargs
		if bayesian_quadrature_kwargs is None:
			bayesian_quadrature_kwargs = {}

		observed_data = self.specification.observed_data

		def target_function(X: np.ndarray) -> np.ndarray:
			parameters = []
			for theta in X:
				parameter_set = {}
				for i, parameter_value in enumerate(theta):
					parameter_name = self.names[i]
					parameter_set[parameter_name] = parameter_value
				parameters.append(parameter_set)

			simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]
			if self.specification.vectorize:
				results = self.calibration_func(
					parameters,
					simulation_ids,
					observed_data,
					**bayesian_quadrature_kwargs,
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**bayesian_quadrature_kwargs,
					)

					results.append(result)
			results = np.array(results)
			return results

		n_init = self.specification.n_init
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		design = RandomDesign(self.parameter_space)
		X = design.get_samples(n_init)
		print(X.shape)
		Y = target_function(X)

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
		task, time_now, outdir = self.prepare_analyze()

		integral_mean, integral_variance = self.quadrature_loop.model.integrate()
		fig, ax = plt.subplots(figsize=self.specification.figsize)
		ax.set_title("Integral density")
		self.rng = np.random.default_rng(self.specification.random_seed)
		integral_samples = self.rng.normal(
			integral_mean, integral_variance, size=self.specification.n_samples
		)
		ax.hist(
			integral_samples,
			alpha=0.5,
		)
		ax.legend()
		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_integral_density.png")
			fig.savefig(outfile)
		else:
			fig.show()

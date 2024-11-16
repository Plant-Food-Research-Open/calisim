"""Contains the implementations for experimental design methods using Emukit

Implements the supported experimental design methods using the Emukit library.

"""

import os.path as osp

import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.experimental_design.acquisitions import (
	IntegratedVarianceReduction,
	ModelVariance,
)
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class EmukitExperimentalDesign(CalibrationWorkflowBase):
	"""The Emukit experimental design method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""

		parameters = []
		self.names = []
		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)
			lower_bound, upper_bound = self.get_parameter_bounds(spec)
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
		experimental_design_kwargs = self.get_calibration_func_kwargs()
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
			if self.specification.batched:
				results = self.calibration_func(
					parameters,
					simulation_ids,
					observed_data,
					**experimental_design_kwargs,
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**experimental_design_kwargs,
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
		Y = target_function(X)

		gp = GPRegression(X, Y, **method_kwargs)
		emulator = GPyModelWrapper(gp)

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
		acquisition = acquisition_class(model=emulator)

		design_loop = ExperimentalDesignLoop(
			model=emulator,
			space=self.parameter_space,
			acquisition=acquisition,
			batch_size=1,
		)
		n_iterations = self.specification.n_iterations
		design_loop.run_loop(target_function, stopping_condition=n_iterations)

		self.emulator = emulator
		self.design_loop = design_loop

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()

		design = RandomDesign(self.parameter_space)
		n_samples = self.specification.n_samples
		X_sample = design.get_samples(n_samples)

		predicted_mu, predicted_var = self.design_loop.model.predict(X_sample)

		observed_data = self.specification.observed_data
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		X = np.arange(0, observed_data.shape[0], 1)
		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)

		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		mean_predicted_mu = predicted_mu.mean(axis=0)
		mean_predicted_var = predicted_var.mean(axis=0)
		axes[1].plot(X, mean_predicted_mu)

		for mult, alpha in [(1, 0.6), (2, 0.4), (3, 0.2)]:
			axes[1].fill_between(
				X,
				mean_predicted_mu - mult * np.sqrt(mean_predicted_var),
				mean_predicted_mu + mult * np.sqrt(mean_predicted_var),
				color="C0",
				alpha=alpha,
			)

		axes[1].set_title(f"Emulated {output_label}")

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_ensemble_{output_label}.png")
			fig.savefig(outfile)
		else:
			fig.show()

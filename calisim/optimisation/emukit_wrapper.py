"""Contains the implementations for optimisation methods using Emukit

Implements the supported optimisation methods using the Emukit library.

"""

import os.path as osp

import numpy as np
import pandas as pd
from emukit.bayesian_optimization.acquisitions import (
	ExpectedImprovement,
	NegativeLowerConfidenceBound,
	ProbabilityOfImprovement,
)
from emukit.bayesian_optimization.acquisitions.local_penalization import (
	LocalPenalization,
)
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.model_wrappers import GPyModelWrapper
from GPy.models import GPRegression
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class EmukitOptimisation(CalibrationWorkflowBase):
	"""The Emukit optimisation method class."""

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
		objective_kwargs = self.specification.calibration_func_kwargs
		if objective_kwargs is None:
			objective_kwargs = {}

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
					parameters, simulation_ids, observed_data, **objective_kwargs
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**objective_kwargs,
					)

					if not isinstance(result, list):
						result = [result]
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

		acquisition_name = self.specification.acquisition_func
		acquisition_funcs = dict(
			ei=ExpectedImprovement,
			poi=ProbabilityOfImprovement,
			la=LogAcquisition,
			lp=LocalPenalization,
			nlcb=NegativeLowerConfidenceBound,
		)
		acquisition_func = acquisition_funcs.get(acquisition_name, None)
		if acquisition_func is None:
			raise ValueError(
				f"Unsupported acquisition function: {acquisition_name}.",
				f"Supported acquisition functions are {', '.join(acquisition_funcs)}",
			)
		acquisition = acquisition_func(model=emulator)

		optimisation_loop = BayesianOptimizationLoop(
			model=emulator,
			space=self.parameter_space,
			acquisition=acquisition,
			batch_size=1,
		)

		n_iterations = self.specification.n_iterations
		optimisation_loop.run_loop(target_function, n_iterations)
		self.emulator = emulator
		self.optimisation_loop = optimisation_loop

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()

		results = self.optimisation_loop.get_results()
		self.results = results
		trial_history = results.best_found_value_per_iteration

		fig, ax = plt.subplots(figsize=self.specification.figsize)
		t = np.arange(0, len(trial_history), 1)
		ax.plot(t, trial_history)
		ax.set_title("Optimisation history")

		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(
				outdir, f"{time_now}-{task}_plot_optimization_history.png"
			)
			fig.savefig(outfile)
		else:
			fig.show()

		if outdir is None:
			return

		optimised_parameters = results.minimum_location
		parameter_dict = {}
		for i, name in enumerate(self.names):
			parameter_dict[name] = optimised_parameters[i]
		parameter_df = pd.DataFrame(parameter_dict, index=[0])
		outfile = osp.join(outdir, f"{time_now}_{task}_parameters.csv")
		parameter_df.to_csv(outfile, index=False)

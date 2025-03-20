"""Contains the implementations for optimisation methods using Emukit

Implements the supported optimisation methods using the Emukit library.

"""

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
from emukit.core.initial_designs import RandomDesign
from matplotlib import pyplot as plt

from ..base import EmukitBase
from ..data_model import ParameterEstimateModel
from ..estimators import EmukitEstimator


class EmukitOptimisation(EmukitBase):
	"""The Emukit optimisation method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		objective_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: np.ndarray) -> np.ndarray:
			return self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				objective_kwargs,
				True,
			)

		n_init = self.specification.n_init
		X, Y = self.get_X_Y(n_init, target_function)
		method_kwargs = self.specification.method_kwargs
		estimator = EmukitEstimator(method_kwargs)
		estimator.fit(X, Y)

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
		acquisition = acquisition_func(model=estimator.emulator)

		optimisation_loop = BayesianOptimizationLoop(
			model=estimator.emulator,
			space=self.parameter_space,
			acquisition=acquisition,
			batch_size=1,
		)

		n_iterations = self.specification.n_iterations
		optimisation_loop.run_loop(target_function, n_iterations)
		self.emulator = estimator
		self.optimisation_loop = optimisation_loop

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		results = self.optimisation_loop.get_results()
		self.results = results
		trial_history = results.best_found_value_per_iteration

		fig, ax = plt.subplots(figsize=self.specification.figsize)
		t = np.arange(0, len(trial_history), 1)
		ax.plot(t, trial_history)
		ax.set_title("Optimisation history")
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, "plot_optimization_history"
		)

		if outdir is None:
			return

		optimised_parameters = results.minimum_location
		parameter_dict = {}
		for i, name in enumerate(self.names):
			parameter_dict[name] = optimised_parameters[i]
		parameter_df = pd.DataFrame(parameter_dict, index=[0])
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}-parameters.csv"
		)
		self.append_artifact(outfile)
		parameter_df.to_csv(outfile, index=False)

		for name in parameter_df.columns:
			estimate = parameter_df[name].item()
			parameter_estimate = ParameterEstimateModel(name=name, estimate=estimate)
			self.add_parameter_estimate(parameter_estimate)

		if self.specification.use_shap and outdir is not None:
			design = RandomDesign(self.parameter_space)
			n_samples = self.specification.n_samples
			X_sample = design.get_samples(n_samples)
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

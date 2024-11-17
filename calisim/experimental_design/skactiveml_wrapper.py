"""Contains the implementations for experimental design methods using scikit-activeml

Implements the supported experimental design methods using the scikit-activeml library.

"""

import os.path as osp

import numpy as np
import pandas as pd
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
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

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class SkActiveMLExperimentalDesign(CalibrationWorkflowBase):
	"""The scikit-activeml experimental design method class."""

	def extend_X(self, X: np.ndarray) -> np.ndarray:
		"""Extend the number of rows for X with a dummy index column.

		Args:
		    X (np.ndarray): The input matrix.

		Returns:
		    np.ndarray: The extended input matrix with a dummy column.
		"""
		design_list = []
		for i in range(X.shape[0]):
			for j in range(self.Y_shape):
				row = np.append(X[i], j)
				design_list.append(row)
		X = np.array(design_list)
		return X

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
				results = self.call_calibration_func(
					parameters,
					simulation_ids,
					observed_data,
					**experimental_design_kwargs,
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.call_calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**experimental_design_kwargs,
					)

					results.append(result)  # type: ignore[arg-type]
			results = np.array(results)
			return results

		n_init = self.specification.n_init
		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		design = RandomDesign(self.parameter_space)
		X = self.specification.X
		if X is None:
			X = design.get_samples(n_init)
		Y_true = self.specification.Y
		if Y_true is None:
			Y_true = target_function(X)

		self.Y_shape = 1
		if len(Y_true.shape) > 1:
			self.Y_shape = Y_true.shape[1]
			if self.Y_shape > 1:
				X = self.extend_X(X)
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
		task, time_now, outdir = self.prepare_analyze()

		design = RandomDesign(self.parameter_space)
		n_samples = self.specification.n_samples
		X_sample = design.get_samples(n_samples)
		if self.Y_shape > 1:
			X_sample = self.extend_X(X_sample)
		predicated = self.emulator.predict(X_sample)

		names = self.names.copy()
		output_label = self.specification.output_labels[0]  # type: ignore[index]
		if X_sample.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(X_sample, columns=names)
		df[f"emulated_{output_label}"] = predicated

		fig, axes = plt.subplots(
			nrows=len(self.names), figsize=self.specification.figsize
		)
		for i, name in enumerate(self.names):
			df.plot.scatter(name, f"emulated_{output_label}", ax=axes[i])
		fig.tight_layout()
		if outdir is not None:
			outfile = osp.join(outdir, f"{time_now}-{task}_emulated_{output_label}.png")
			fig.savefig(outfile)
		else:
			fig.show()

		if outdir is None:
			return

		outfile = osp.join(outdir, f"{time_now}_{task}_emulated_{output_label}.csv")
		df.to_csv(outfile, index=False)

"""Contains the implementations for evolutionary algorithms using EvoTorch

Implements the supported evolutionary algorithms using the EvoTorch library.

"""

from typing import Any

import evotorch.algorithms as algos
import evotorch.operators as operators
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from evotorch import Problem
from evotorch.logging import PandasLogger
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType, ParameterEstimateModel


class EvoTorchEvolutionary(CalibrationWorkflowBase):
	"""The EvoTorch evolutionary algorithm method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		self.bounds = []
		self.lower_bounds = []
		self.upper_bounds = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			bounds = self.get_parameter_bounds(spec)
			self.bounds.append(bounds)

			lower_bound, upper_bound = bounds
			if data_type == ParameterDataType.DISCRETE:
				lower_bound = np.floor(lower_bound).astype("int")
				upper_bound = np.floor(upper_bound).astype("int")
			self.lower_bounds.append(lower_bound)
			self.upper_bounds.append(upper_bound)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		directions = []
		for direction in self.specification.directions:
			if direction == "minimize":
				direction = "min"
			elif direction == "maximize":
				direction = "max"
			directions.append(direction)

		self.trials: list[dict[str, Any]] = []
		evolutionary_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: torch.Tensor) -> torch.Tensor:
			trial = {}
			if X.ndim == 1:
				for i, name in enumerate(self.names):
					trial[name] = X[i].item()
					self.trials.append(trial)
				X = [X]

			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				evolutionary_kwargs,
				False,
			)

			self.set_output_labels_from_Y(Y)
			output_labels = self.specification.output_labels
			for i, label in enumerate(output_labels):  # type: ignore[arg-type]
				trial[label] = Y[i]
			return Y

		vectorized = self.specification.batched
		seed = self.specification.random_seed
		self.problem = Problem(
			directions,
			target_function,
			solution_length=len(self.names),
			bounds=(self.lower_bounds, self.upper_bounds),
			vectorized=vectorized,
			seed=seed,
		)

		algorithm_name = self.specification.method
		supported_algorithms = dict(
			ga=algos.GeneticAlgorithm, cosyne=algos.Cosyne, cmaes=algos.PyCMAES
		)
		algorithm_class = supported_algorithms.get(algorithm_name, None)
		if algorithm_class is None:
			raise ValueError(
				f"Unsupported EvoTorch algorithm: {algorithm_name}.",
				f"Supported EvoTorch algorithms are {', '.join(supported_algorithms)}",
			)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		if algorithm_name == "cmaes":
			method_kwargs["obj_index"] = 0
		elif algorithm_name == "ga":
			operator_dict = self.specification.operators
			operator_list = []
			if operator_dict is not None:
				for operator_name, operator_kwargs in operator_dict.items():
					operator_class = getattr(operators, operator_name)
					operator = operator_class(self.problem, **operator_kwargs)
					operator_list.append(operator)
			method_kwargs["operators"] = operator_list

		self.searcher = algorithm_class(self.problem, **method_kwargs)
		interval = self.specification.n_samples
		self.logger = PandasLogger(self.searcher, interval=interval)

		num_generations = self.specification.n_iterations
		self.searcher.run(num_generations=num_generations)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		trials_df = pd.DataFrame(self.trials)
		objective_df = self.logger.to_dataframe()

		output_labels = self.specification.output_labels
		trials_df_best = trials_df.sort_values(output_labels).head(1)
		for name in self.names:
			estimate = trials_df_best[name].item()
			parameter_estimate = ParameterEstimateModel(name=name, estimate=estimate)
			self.add_parameter_estimate(parameter_estimate)

		fig, axes = plt.subplots(
			nrows=len(output_labels),  # type: ignore[arg-type]
			figsize=self.specification.figsize,
		)

		if not isinstance(axes, list):
			axes = [axes]

		trials_df["index"] = trials_df.index
		for i, output_label in enumerate(output_labels):  # type: ignore[arg-type]
			trials_df.plot.scatter(
				"index", output_label, ax=axes[i], title=f"Simulated {output_label}"
			)
		trials_df = trials_df.drop(columns="index")

		self.present_fig(fig, outdir, time_now, task, experiment_name, "trial-history")

		parameter_names = self.names
		for output_label in output_labels:
			fig = make_subplots(
				rows=1, cols=len(parameter_names), subplot_titles=parameter_names
			)
			for i, parameter_name in enumerate(parameter_names):
				fig.add_trace(
					go.Scatter(
						x=trials_df[parameter_name],
						y=trials_df[output_label],
						mode="markers",
					),
					row=1,
					col=i + 1,
				)

			fig.update_layout(yaxis_title=output_label, showlegend=False)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}-{output_label}-plot_slice.png",
				)
				self.append_artifact(outfile)
				fig.write_image(outfile)
			else:
				fig.show()

		if outdir is None:
			return

		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}-objective.csv"
		)
		self.append_artifact(outfile)
		objective_df.to_csv(outfile, index=False)

		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}-trials.csv")
		self.append_artifact(outfile)
		trials_df.to_csv(outfile, index=False)

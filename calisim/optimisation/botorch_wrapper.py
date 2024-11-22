"""Contains the implementations for optimisation methods using BoTorch

Implements Bayesian optimisation methods using the BoTorch library.

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax import Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from plotly.subplots import make_subplots

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType


class BoTorchOptimisation(CalibrationWorkflowBase):
	"""The BoTorchOptimisation optimisation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		parameters = []
		parameter_names = []
		data_types = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			name = spec.name
			parameter_names.append(name)

			lower, upper = self.get_parameter_bounds(spec)

			data_type = spec.data_type
			data_types.append(data_type)
			if data_type == ParameterDataType.CONTINUOUS:
				parameter_type = ParameterType.FLOAT
			else:
				parameter_type = ParameterType.INT

			parameters.append(
				RangeParameter(
					name=name, parameter_type=parameter_type, lower=lower, upper=upper
				)
			)
		search_space = SearchSpace(parameters=parameters)
		objective_kwargs = self.get_calibration_func_kwargs()

		workflow = self

		class ObjectiveMetric(NoisyFunctionMetric):
			def f(
				self, x: np.ndarray
			) -> float | list[float] | np.ndarray | pd.DataFrame:
				X = [x]
				results = workflow.calibration_func_wrapper(
					X,
					workflow,
					workflow.specification.observed_data,
					parameter_names,
					data_types,
					objective_kwargs,
				)
				return results[0]

		optimization_config = OptimizationConfig(
			objective=Objective(
				metric=ObjectiveMetric(
					name=f"{self.specification.experiment_name}_metric",
					param_names=parameter_names,
					noise_sd=None,
				),
				minimize=self.specification.directions[0] == "minimize",
			)
		)

		self.experiment = Experiment(
			name=self.specification.experiment_name,
			search_space=search_space,
			optimization_config=optimization_config,
			runner=SyntheticRunner(),
		)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		experiment = self.experiment

		sobol = Models.SOBOL(search_space=experiment.search_space)
		n_init = self.specification.n_init
		n_trials = self.specification.n_samples

		for i in range(n_init):
			generator_run = sobol.gen(n=1)
			trial = experiment.new_trial(generator_run=generator_run)
			trial.run()
			trial.mark_completed()

		for i in range(n_trials):
			if self.specification.verbose:
				print(f"Running BO trial {i + n_init + 1}/{n_init + n_trials}")
			gpei = Models.BOTORCH_MODULAR(
				experiment=experiment, data=experiment.fetch_data()
			)
			generator_run = gpei.gen(n=1)
			trial = experiment.new_trial(generator_run=generator_run)
			trial.run()
			trial.mark_completed()

		self.trial = trial  # type: ignore[possibly-undefined]

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()

		trials = []
		for trial in self.experiment.trials.values():
			if isinstance(trial.arm, list):
				for arm in trial.arm:
					parameters = arm.parameters
					parameters["arm_name"] = arm.name
					trials.append(parameters)
			else:
				parameters = trial.arm.parameters
				parameters = {f"param_{k}": parameters[k] for k in parameters}
				parameters["arm_name"] = trial.arm.name
				trials.append(parameters)

		trials_df = pd.DataFrame(trials).set_index("arm_name")
		objective_df = self.experiment.fetch_data().df.set_index("arm_name")
		trials_df = (
			trials_df.join(objective_df)
			.reset_index()
			.sort_values("mean", ascending=True)
		)

		if outdir is not None:
			outfile = self.join(outdir, f"{time_now}_{task}_objective.csv")
			trials_df.to_csv(outfile, index=False)

		parameter_names = [col for col in trials_df if col.startswith("param_")]
		fig = make_subplots(
			rows=1, cols=len(parameter_names), subplot_titles=parameter_names
		)
		for i, parameter_name in enumerate(parameter_names):
			fig.add_trace(
				go.Scatter(x=trials_df[parameter_name], y=trials_df["mean"]),
				row=1,
				col=i + 1,
			)

		fig.update_layout(yaxis_title="Score", showlegend=False)
		if outdir is not None:
			outfile = self.join(outdir, f"{time_now}_{task}_slice_plot.png")
			fig.write_image(outfile)
		else:
			fig.show()

		fig = go.Figure(
			data=go.Scatter(x=trials_df["trial_index"], y=trials_df["mean"])
		)
		fig.update_layout(xaxis_title="Trial", yaxis_title="Score", showlegend=False)
		if outdir is not None:
			outfile = self.join(outdir, f"{time_now}_{task}_trial_history.png")
			fig.write_image(outfile)
		else:
			fig.show()

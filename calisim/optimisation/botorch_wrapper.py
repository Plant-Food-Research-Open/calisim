"""Contains the implementations for optimisation methods using BoTorch

Implements Bayesian optimisation methods using the BoTorch library.

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.plot.base import AxPlotConfig
from ax.plot.render import plot_config_to_html
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.utils.report.render import render_report_elements
from plotly.subplots import make_subplots

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType, ParameterEstimateModel


class BoTorchOptimisation(CalibrationWorkflowBase):
	"""The BoTorchOptimisation optimisation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []

		parameter_spec = self.specification.parameter_spec.parameters
		parameters = []
		objectives = {}

		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)
			if data_type == ParameterDataType.DISCRETE:
				value_type = "int"
			else:
				value_type = "float"

			bounds = self.get_parameter_bounds(spec)

			parameter = dict(
				name=parameter_name,
				type="range",
				bounds=list(bounds),
				value_type=value_type,
			)
			parameters.append(parameter)

		n_out = self.specification.n_out
		self.output_labels = self.specification.output_labels
		directions = self.specification.directions
		if self.output_labels is None:
			self.output_labels = [f"objective_{i}" for i in range(n_out)]

		for i, output_label in enumerate(self.output_labels):
			direction = directions[i]
			if direction == "minimize":
				minimize = True
			else:
				minimize = False
			objectives[output_label] = ObjectiveProperties(minimize=minimize)

		self.ax_client = AxClient(verbose_logging=False)
		self.ax_client.create_experiment(
			name=self.specification.experiment_name,
			parameters=parameters,
			objectives=objectives,
			overwrite_existing_experiment=True,
		)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		objective_kwargs = self.get_calibration_func_kwargs()

		def target_function(X: dict[str, float]) -> np.ndarray:
			X = list(X.values())
			X = [X]

			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				objective_kwargs,
				True,
			)
			Y = Y[0]

			results = {}
			for i, output in enumerate(self.output_labels):  # type: ignore[arg-type]
				results[output] = (Y[i], 0.0)

			return results

		n_iterations = self.specification.n_iterations
		for _ in range(n_iterations):
			parameterization, trial_index = self.ax_client.get_next_trial()
			self.ax_client.complete_trial(
				trial_index=trial_index, raw_data=target_function(parameterization)
			)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		def plot_config(config: AxPlotConfig, title: str) -> None:
			fig = plot_config_to_html(config)
			outfile = self.join(
				outdir,  # type: ignore[arg-type]
				f"{time_now}-{task}-{experiment_name}-{title}.html",
			)

			if outdir is not None:
				with open(outfile, "w") as f:
					f.write(
						render_report_elements(
							title,
							html_elements=[fig],
							header=False,
						)
					)
				self.append_artifact(outfile)
			else:
				render(config)

		config = self.ax_client.get_optimization_trace()
		plot_config(config, "optimization-trace")
		config = self.ax_client.get_feature_importances()
		plot_config(config, "feature-importances")

		trials = []
		for trial in self.ax_client.experiment.trials.values():
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
		objective_df = self.ax_client.experiment.fetch_data().df.set_index("arm_name")
		trials_df = (
			trials_df.join(objective_df)
			.reset_index()
			.sort_values("mean", ascending=True)
		)

		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-objective.csv"
			)
			self.append_artifact(outfile)
			trials_df.to_csv(outfile, index=False)

		parameter_names = [col for col in trials_df if col.startswith("param_")]
		fig = make_subplots(
			rows=1, cols=len(parameter_names), subplot_titles=parameter_names
		)
		for i, parameter_name in enumerate(parameter_names):
			fig.add_trace(
				go.Scatter(
					x=trials_df[parameter_name], y=trials_df["mean"], mode="markers"
				),
				row=1,
				col=i + 1,
			)

		fig.update_layout(yaxis_title="Score", showlegend=False)
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-plot_slice.png"
			)
			self.append_artifact(outfile)
			fig.write_image(outfile)
		else:
			fig.show()

		fig = go.Figure(
			data=go.Scatter(
				x=trials_df["trial_index"], y=trials_df["mean"], mode="markers"
			)
		)
		fig.update_layout(xaxis_title="Trial", yaxis_title="Score", showlegend=False)
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-trial_history.png"
			)
			fig.write_image(outfile)
			self.append_artifact(outfile)
		else:
			fig.show()

		values = [value for value in trials_df.columns if value.startswith("mean")]

		trials_df_best = trials_df.sort_values(values).head(1)
		for col in trials_df_best.columns:
			if not col.startswith("param_"):
				continue
			name = col.replace("param_", "")
			estimate = trials_df_best[col].item()
			parameter_estimate = ParameterEstimateModel(name=name, estimate=estimate)
			self.add_parameter_estimate(parameter_estimate)

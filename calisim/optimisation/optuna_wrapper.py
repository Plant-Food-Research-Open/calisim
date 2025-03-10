"""Contains the implementations for optimisation methods using Optuna

Implements the supported optimisation methods using the Optuna library.

"""

from collections.abc import Callable

import numpy as np
import optuna
import optuna.samplers as opt_samplers
import pandas as pd

from ..base import CalibrationWorkflowBase
from ..data_model import DistributionModel, ParameterDataType, ParameterEstimateModel


class OptunaOptimisation(CalibrationWorkflowBase):
	"""The Optuna optimisation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		sampler_name = self.specification.method
		supported_samplers = dict(
			tpes=opt_samplers.TPESampler,
			cmaes=opt_samplers.CmaEsSampler,
			nsga=opt_samplers.NSGAIISampler,
			qmc=opt_samplers.QMCSampler,
			gp=opt_samplers.GPSampler,
		)
		sampler_class = supported_samplers.get(sampler_name, None)
		if sampler_class is None:
			raise ValueError(
				f"Unsupported Optuna sampler: {sampler_name}.",
				f"Supported Optuna samplers are {', '.join(supported_samplers)}",
			)
		sampler_kwargs = self.specification.method_kwargs
		if sampler_kwargs is None:
			sampler_kwargs = {}
		self.sampler = sampler_class(**sampler_kwargs)

		self.study = optuna.create_study(
			sampler=self.sampler,
			study_name=self.specification.experiment_name,
			directions=self.specification.directions,
		)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		parameter_spec = self.specification.parameter_spec.parameters

		def objective(
			trial: optuna.trial.Trial,
			parameter_spec: list[DistributionModel],
			observed_data: np.ndarray | pd.DataFrame,
			call_calibration_func: Callable,
			objective_kwargs: dict,
		) -> float | list[float]:
			parameters = {}
			for spec in parameter_spec:
				parameter_name = spec.name
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				data_type = spec.data_type

				if data_type == ParameterDataType.CONTINUOUS:
					parameters[parameter_name] = trial.suggest_float(
						parameter_name, lower_bound, upper_bound
					)
				else:
					parameters[parameter_name] = trial.suggest_int(
						parameter_name, lower_bound, upper_bound
					)

			simulation_id = self.get_simulation_uuid()
			return call_calibration_func(
				parameters, simulation_id, observed_data, **objective_kwargs
			)

		objective_kwargs = self.get_calibration_func_kwargs()

		self.study.optimize(
			lambda trial: objective(
				trial,
				parameter_spec,  # type: ignore[arg-type]
				self.specification.observed_data,
				call_calibration_func=self.call_calibration_func,
				objective_kwargs=objective_kwargs,
			),
			n_trials=self.specification.n_iterations,
			n_jobs=self.specification.n_jobs,
		)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		for plot_func in [
			optuna.visualization.plot_edf,
			optuna.visualization.plot_optimization_history,
			optuna.visualization.plot_parallel_coordinate,
			optuna.visualization.plot_param_importances,
			optuna.visualization.plot_slice,
		]:
			optimisation_plot = plot_func(self.study)
			if outdir is not None:
				outfile = self.join(
					outdir,
					f"{time_now}-{task}-{experiment_name}_{plot_func.__name__}.png",
				)
				self.append_artifact(outfile)
				optimisation_plot.write_image(outfile)
			else:
				optimisation_plot.show()

		if outdir is None:
			return

		trials_df: pd.DataFrame = self.study.trials_dataframe()
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trials.csv")
		self.append_artifact(outfile)
		trials_df.to_csv(outfile, index=False)

		values = [value for value in trials_df.columns if value.startswith("value")]

		trials_df_best = trials_df.sort_values(values).head(1)
		for col in trials_df_best.columns:
			if not col.startswith("params_"):
				continue
			name = col.replace("params_", "")
			estimate = trials_df_best[col].item()
			parameter_estimate = ParameterEstimateModel(name=name, estimate=estimate)
			self.add_parameter_estimate(parameter_estimate)

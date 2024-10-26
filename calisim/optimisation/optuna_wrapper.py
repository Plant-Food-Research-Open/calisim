"""Contains the implementations for optimisation methods using Optuna

Implements the supported optimisation methods using the Optuna library.

"""

import os.path as osp
from collections.abc import Callable

import numpy as np
import optuna
import optuna.samplers as opt_samplers
import pandas as pd

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_datetime_now


class OptunaOptimisation(CalibrationWorkflowBase):
	"""The Optuna optimisation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		sampler_name = self.specification.sampler
		supported_samplers = dict(
			tpes=opt_samplers.TPESampler,
			cmaes=opt_samplers.CmaEsSampler,
			nsga=opt_samplers.NSGAIISampler,
			qmc=opt_samplers.QMCSampler,
		)
		sampler_class = supported_samplers.get(sampler_name, None)
		if sampler_class is None:
			raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")
		self.sampler = sampler_class(**self.specification.sampler_kwargs)

		self.study = optuna.create_study(
			sampler=self.sampler,
			study_name=self.specification.experiment_name,
			directions=self.specification.directions,
		)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		parameter_spec = self.specification.parameter_spec

		def objective(
			trial: optuna.trial.Trial,
			parameter_spec: list,
			observed_data: np.ndarray | pd.DataFrame,
			objective_func: Callable,
			objective_kwargs: dict,
		) -> float | list[float]:
			parameters = {}
			for spec in parameter_spec:
				parameter_name = spec.name
				lower_bound = spec.lower_bound
				upper_bound = spec.upper_bound
				data_type = spec.data_type

				if data_type == ParameterDataType.CONTINUOUS:
					parameters[parameter_name] = trial.suggest_float(
						parameter_name, lower_bound, upper_bound
					)
				else:
					parameters[parameter_name] = trial.suggest_int(
						parameter_name, lower_bound, upper_bound
					)

			return objective_func(parameters, observed_data, **objective_kwargs)

		objective_kwargs = self.specification.calibration_kwargs
		if objective_kwargs is None:
			objective_kwargs = {}

		self.study.optimize(
			lambda trial: objective(
				trial,
				parameter_spec,
				self.specification.observed_data,
				objective_func=self.calibration_func,
				objective_kwargs=objective_kwargs,
			),
			n_trials=self.specification.n_samples,
			n_jobs=self.specification.n_jobs,
		)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task = "optimisation"
		time_now = get_datetime_now()
		outdir = self.specification.outdir

		for plot_func in [
			optuna.visualization.plot_edf,
			optuna.visualization.plot_optimization_history,
			optuna.visualization.plot_parallel_coordinate,
			optuna.visualization.plot_param_importances,
			optuna.visualization.plot_slice,
		]:
			optimisation_plot = plot_func(self.study)
			if outdir is not None:
				outfile = osp.join(
					outdir, f"{time_now}_{task}_{plot_func.__name__}.png"
				)
				optimisation_plot.write_image(outfile)
			else:
				optimisation_plot.show()

		if outdir is None:
			return

		trials_df: pd.DataFrame = self.study.trials_dataframe().sort_values(
			"value", ascending=True
		)
		outfile = osp.join(outdir, f"{time_now}_{task}_trials.csv")
		trials_df.to_csv(outfile, index=False)

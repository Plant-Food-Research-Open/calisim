"""Contains the implementations for sensitivity analysis methods using SALib

Implements the supported sensitivity analysis methods using the SALib library.

"""

import os.path as osp
from pydoc import locate

import numpy as np
import plotly.express as px
from SALib import ProblemSpec

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_datetime_now


class SALibSensitivityAnalysis(CalibrationWorkflowBase):
	"""The Optuna optimisation method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		bounds = []
		dists = []

		parameter_spec = self.specification.parameter_spec
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)
			dists.append("unif")
			lower_bound = spec.lower_bound
			upper_bound = spec.upper_bound
			bounds.append([lower_bound, upper_bound])

		problem = {
			"num_vars": len(self.names),
			"names": self.names,
			"bounds": bounds,
			"dists": dists,
			"groups": None,
			"outputs": self.specification.output_labels,
		}

		self.sp = ProblemSpec(problem)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		sampler_name = self.specification.sampler
		n_samples = self.specification.n_samples

		sample_func_name = f"SALib.sample.{sampler_name}.sample"
		sample_func = locate(sample_func_name)
		if sample_func is None:
			raise ValueError(f"Sampler not implemented in SALib: {sampler_name}")
		self.parameter_values = sample_func(  # type: ignore[operator]
			self.sp, n_samples, seed=self.specification.random_seed
		)

		data_types = []
		parameter_spec = self.specification.parameter_spec
		for spec in parameter_spec:
			data_type = spec.data_type
			data_types.append(data_type)

		parameters = []
		for x in self.parameter_values:
			theta = {}
			for i, name in enumerate(self.names):
				data_type = data_types[i]
				if data_type == ParameterDataType.CONTINUOUS:
					theta[name] = x[i].item()
				else:
					theta[name] = int(x[i].item())
			parameters.append(theta)

		sensitivity_kwargs = self.specification.calibration_kwargs
		if sensitivity_kwargs is None:
			sensitivity_kwargs = {}

		if self.specification.vectorize:
			self.results = self.calibration_func(
				parameters, self.specification.observed_data, **sensitivity_kwargs
			)
		else:
			self.results = []
			for parameter in parameters:
				result = self.calibration_func(
					parameter, self.specification.observed_data, **sensitivity_kwargs
				)
				self.results.append(result)

		self.results = np.array(self.results)

		analyze_func_name = f"SALib.analyze.{sampler_name}.analyze"
		analyze_func = locate(analyze_func_name)
		if analyze_func is None:
			raise ValueError(
				f"Analysis for sampler not implemented in SALib: {sampler_name}"
			)

		analyze_kwargs = self.specification.sampler_kwargs
		analyze_kwargs["seed"] = self.specification.random_seed  # type: ignore[index]
		self.sensitivity_indices = analyze_func(self.sp, self.results, **analyze_kwargs)  # type: ignore[operator]

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task = "sensitivity_analysis"
		time_now = get_datetime_now()
		outdir = self.specification.outdir

		si_dfs = self.sensitivity_indices.to_df()

		for i in range(len(si_dfs)):
			si_df = si_dfs[i].reset_index()
			si_df["index"] = si_df["index"].astype("str")
			si_type = si_df.columns[1]
			fig = px.bar(
				si_df,
				x="index",
				y=si_type,
				error_y=f"{si_type}_conf",
			).update_layout(xaxis_title="Parameter", yaxis_title=si_type)

			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}_{task}_{si_type}.png")
				fig.write_image(outfile)
			else:
				fig.show()

		if outdir is None:
			return

		for si_df in si_dfs:
			si_df = si_df.reset_index().rename(columns={"index": "parameter"})

			si_type = si_df.columns[1]
			outfile = osp.join(outdir, f"{time_now}_{task}_{si_type}.csv")
			si_df.to_csv(outfile, index=False)

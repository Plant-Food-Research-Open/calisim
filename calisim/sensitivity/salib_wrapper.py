"""Contains the implementations for sensitivity analysis methods using SALib

Implements the supported sensitivity analysis methods using the SALib library.

"""

import pandas as pd
from matplotlib import pyplot as plt
from SALib import ProblemSpec

from ..base import CalibrationWorkflowBase


class SALibSensitivityAnalysis(CalibrationWorkflowBase):
	"""The SALib sensitivity analysis method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		bounds = []
		dists = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			dists.append("unif")

			lower_bound, upper_bound = self.get_parameter_bounds(spec)
			bounds.append([lower_bound, upper_bound])

		problem = {
			"num_vars": len(self.names),
			"names": self.names,
			"bounds": bounds,
			"dists": dists,
			"groups": self.specification.groups,
			"outputs": self.specification.output_labels,
		}

		self.sp = ProblemSpec(problem)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		data_types = []
		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			data_type = spec.data_type
			data_types.append(data_type)

		sampler_name = self.specification.method
		sample_func = getattr(self.sp, f"sample_{sampler_name}")
		sampler_kwargs = self.specification.method_kwargs
		if sampler_kwargs is None:
			sampler_kwargs = {}
		sampler_kwargs["seed"] = self.specification.random_seed
		n_samples = self.specification.n_samples

		sp_samples = self.specification.X
		if sp_samples is None:
			sample_func(n_samples, **sampler_kwargs)
		else:
			self.sp.samples = sp_samples

		sensitivity_kwargs = self.get_calibration_func_kwargs()
		sp_results = self.specification.Y
		if sp_results is None:
			self.sp.evaluate(
				self.calibration_func_wrapper,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				sensitivity_kwargs,
			)
		else:
			self.sp.results = sp_results

		analyze_func = getattr(self.sp, f"analyze_{sampler_name}")
		analyze_kwargs = self.specification.analyze_kwargs
		if analyze_kwargs is None:
			analyze_kwargs = {}
		analyze_kwargs["seed"] = self.specification.random_seed
		analyze_func(**analyze_kwargs)

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		sampler_name = self.specification.method

		self.sp.plot()
		plt.tight_layout()
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-indices.png"
			)
			self.append_artifact(outfile)
			plt.savefig(outfile)
		else:
			plt.show()
		plt.close()

		self.sp.heatmap()
		plt.tight_layout()
		if outdir is not None:
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-heatmap.png"
			)
			self.append_artifact(outfile)
			plt.savefig(outfile)
		else:
			plt.show()
		plt.close()

		if outdir is None:
			return

		def recursive_write_csv(dfs: pd.DataFrame) -> None:
			if isinstance(dfs, list):
				for df in dfs:
					recursive_write_csv(df)
			else:
				si_df = dfs.reset_index().rename(columns={"index": "parameter"})
				si_type = si_df.columns[1]
				outfile = self.join(
					outdir, f"{time_now}-{task}-{experiment_name}-{si_type}.csv"
				)
				self.append_artifact(outfile)
				si_df.to_csv(outfile, index=False)

		si_dfs = self.sp.to_df()
		if isinstance(si_dfs, list):
			recursive_write_csv(si_dfs)
		else:
			si_df = si_dfs.reset_index().rename(columns={"index": "parameter"})
			outfile = self.join(
				outdir, f"{time_now}-{task}-{experiment_name}-{sampler_name}.csv"
			)
			self.append_artifact(outfile)
			si_df.to_csv(outfile, index=False)

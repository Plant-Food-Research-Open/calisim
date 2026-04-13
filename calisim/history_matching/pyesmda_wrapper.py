"""Contains the implementations for history matching methods using
pyESMDA

Implements the supported history matching methods using
the pyESMDA library.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyesmda import ESMDA, ESMDA_RS

from ..base import HistoryMatchingBase
from ..data_model import ParameterEstimateModel


def forward_model(_: np.ndarray, workflow: HistoryMatchingBase) -> np.ndarray:
	"""The forward model for the ensemble simulation.

	Args:
		m_ensemble (np.ndarray): The ensemble simulation parameters.
		workflow (HistoryMatchingBase) The calibration workflow.

	Returns:
		np.ndarray: The ensemble results.
	"""
	parameter_spec = workflow.parameters
	X = pd.DataFrame(parameter_spec).values
	ensemble_outputs = workflow.calibration_func_wrapper(
		X,
		workflow,
		workflow.specification.observed_data,
		workflow.names,
		workflow.data_types,
		workflow.get_calibration_func_kwargs(),
	)
	return ensemble_outputs


class PyESMDAHistoryMatching(HistoryMatchingBase):
	"""The pyESMDA history matching method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		smoother_name = self.specification.method
		smoothers = dict(esmda=ESMDA, esmda_rs=ESMDA_RS)
		smoother_class = smoothers.get(smoother_name, None)
		if smoother_class is None:
			raise ValueError(
				f"Unsupported ensemble smoother: {smoother_name}.",
				f"Supported ensemble smoothers are {', '.join(smoothers)}",
			)

		n_jobs = self.specification.n_jobs
		if n_jobs > 1:
			is_parallel_analyse_step = True
		else:
			is_parallel_analyse_step = False

		observed_data = self.specification.observed_data
		cov_obs = self.specification.covariance
		if cov_obs is None:
			cov_obs = np.eye(observed_data.shape[0])

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		m_init = self.specification.X
		if m_init is None:
			m_init = [self.parameters[k] for k in self.parameters]  # type: ignore[index]
			m_init = np.array(m_init).T

		if smoother_name == "esmda":
			method_kwargs["n_assimilations"] = self.specification.n_iterations

		self.solver = smoother_class(
			obs=observed_data,
			m_init=m_init,
			forward_model=forward_model,
			forward_model_kwargs=dict(workflow=self),
			cov_obs=cov_obs,
			random_state=self.specification.random_seed,
			batch_size=n_jobs,
			is_parallel_analyse_step=is_parallel_analyse_step,
			**method_kwargs,
		)

		self.solver.solve()

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		smoother_name = self.specification.method
		n_iterations = self.specification.n_iterations
		n_samples = self.specification.n_samples

		means = np.average(self.solver.m_prior, axis=0)
		stds = np.sqrt(np.diagonal(self.solver.cov_mm))

		parameter_names = list(self.parameters.keys())
		parameter_samples = {}

		fig, axes = plt.subplots(
			nrows=len(parameter_names), figsize=self.specification.figsize
		)
		if not isinstance(axes, np.ndarray):
			axes = [axes]

		for i, parameter_name in enumerate(parameter_names):
			axes[i].set_title(parameter_name)
			axes[i].hist(self.parameters[parameter_name], label="Prior")  # type: ignore[index]
			mu = means[i]
			sigma = stds[i]
			parameter_samples[parameter_name] = self.rng.normal(
				mu, sigma, size=n_samples
			)
			axes[i].hist(
				parameter_samples[parameter_name],
				label=f"{smoother_name} ({n_iterations}) Posterior",
				alpha=0.5,
			)
			axes[i].legend()
		self.present_fig(fig, outdir, time_now, task, experiment_name, "plot-slice")

		pred_dfs = [pd.DataFrame(preds) for preds in self.solver.d_pred]
		output_labels = self.specification.output_labels
		if output_labels is None:
			output_labels = ["output"]
		output_label = output_labels[0]

		observed_data = self.specification.observed_data
		fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)
		X = np.arange(0, observed_data.shape[0], 1)
		axes[0].plot(X, observed_data)
		axes[0].set_title(f"Observed {output_label}")

		for pred_df in pred_dfs:
			pred_df[0].plot(ax=axes[1])
		axes[1].set_title(f"Ensemble {output_label}")
		fig.tight_layout()
		self.present_fig(
			fig, outdir, time_now, task, experiment_name, f"ensemble-{output_label}"
		)

		X_IES_df = pd.DataFrame(parameter_samples)
		for name in X_IES_df:
			estimate = X_IES_df[name].mean()
			uncertainty = X_IES_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

		if outdir is None:
			return

		self.to_csv(X_IES_df, "posterior")

		for pred_df in pred_dfs:
			pred_df["x"] = pred_df.index
		pred_df = pd.concat(pred_dfs)
		pred_df.columns = [output_label, "x"]
		self.to_csv(pred_df, f"ensemble-{output_label}")

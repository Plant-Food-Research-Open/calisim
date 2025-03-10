"""Contains the implementations for simulation-based inference methods using
SBI

Implements the supported simulation-based inference methods using
the SBI library.

"""

import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import (
	SNPE,
	prepare_for_sbi,
	simulate_for_sbi,
)

from ..base import SimulationBasedInferenceBase
from ..data_model import ParameterEstimateModel


class SBISimulationBasedInference(SimulationBasedInferenceBase):
	"""The SBI simulation-based inference method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		sbi_kwargs = self.get_calibration_func_kwargs()

		def simulator_func(X: np.ndarray) -> np.ndarray:
			X = X.detach().cpu().numpy()
			X = [X]
			results = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				sbi_kwargs,
			)
			return results[0]

		simulator, prior = prepare_for_sbi(simulator_func, self.parameters)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		embedding_net = nn.Identity()
		neural_posterior = utils.posterior_nn(
			model=self.specification.method,
			embedding_net=embedding_net,
			**method_kwargs,
		)

		inference = SNPE(prior=prior, density_estimator=neural_posterior)

		theta = self.specification.X
		x = self.specification.Y
		if theta is None or x is None:
			theta, x = simulate_for_sbi(
				simulator,
				proposal=prior,
				num_simulations=self.specification.num_simulations,
			)

		inference = inference.append_simulations(theta, x)
		density_estimator = inference.train(
			max_num_epochs=self.specification.n_iterations
		)
		posterior = inference.build_posterior(density_estimator)
		posterior.set_default_x(self.specification.observed_data)

		self.prior = prior
		self.simulator = simulator
		self.inference = inference
		self.posterior = posterior

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		n_draws = self.specification.n_samples
		posterior_samples = self.posterior.sample(
			(n_draws,), x=self.specification.observed_data
		)

		limits = []
		lower_limits, _ = posterior_samples.min(axis=0)
		upper_limits, _ = posterior_samples.max(axis=0)
		for i in range(len(self.names)):
			limits.append((lower_limits[i], upper_limits[i]))

		for plot_func in [analysis.pairplot, analysis.marginal_plot]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(
				posterior_samples, figsize=(24, 24), labels=self.names, limits=limits
			)
			self.present_fig(
				fig, outdir, time_now, task, experiment_name, plot_func.__name__
			)

		for plot_func in [
			analysis.conditional_pairplot,
			analysis.conditional_marginal_plot,
		]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(
				density=self.posterior,
				condition=self.posterior.sample((1,)),
				figsize=(24, 24),
				labels=self.names,
				limits=limits,
			)
			self.present_fig(
				fig, outdir, time_now, task, experiment_name, plot_func.__name__
			)

		thetas = self.prior.sample((n_draws,))
		xs = self.simulator(thetas)
		ranks, dap_samples = analysis.run_sbc(
			thetas, xs, self.posterior, num_posterior_samples=n_draws
		)

		num_bins = None
		if n_draws <= 20:
			num_bins = n_draws

		for plot_type in ["hist", "cdf"]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = analysis.sbc_rank_plot(
				ranks=ranks,
				num_bins=num_bins,
				num_posterior_samples=n_draws,
				plot_type=plot_type,
				parameter_labels=self.names,
			)
			fig_suffix = f"{analysis.sbc_rank_plot.__name__}_{plot_type}"
			self.present_fig(fig, outdir, time_now, task, experiment_name, fig_suffix)

		if outdir is None:
			return

		check_stats = analysis.check_sbc(
			ranks, thetas, dap_samples, num_posterior_samples=n_draws
		)

		check_stats_list = []
		for metric in check_stats:
			metric_dict = {"metric": metric}
			check_stats_list.append(metric_dict)
			scores = check_stats[metric].detach().cpu().numpy()
			for i, score in enumerate(scores):
				col_name = self.names[i]
				metric_dict[col_name] = score

		check_stats_df = pd.DataFrame(check_stats_list)
		outfile = self.join(
			outdir, f"{time_now}-{task}-{experiment_name}_diagnostics.csv"
		)
		self.append_artifact(outfile)
		check_stats_df.to_csv(outfile, index=False)

		trace_df = pd.DataFrame(
			posterior_samples.cpu().detach().numpy(), columns=self.names
		)
		outfile = self.join(outdir, f"{time_now}-{task}-{experiment_name}_trace.csv")
		self.append_artifact(outfile)
		trace_df.to_csv(outfile, index=False)

		for name in trace_df:
			estimate = trace_df[name].mean()
			uncertainty = trace_df[name].std()

			parameter_estimate = ParameterEstimateModel(
				name=name, estimate=estimate, uncertainty=uncertainty
			)
			self.add_parameter_estimate(parameter_estimate)

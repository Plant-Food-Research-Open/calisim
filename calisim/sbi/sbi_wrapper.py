"""Contains the implementations for simulation-based inference methods using
SBI

Implements the supported simulation-based inference methods using
the SBI library.

"""

import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
from matplotlib import pyplot as plt
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import (
	SNPE,
	prepare_for_sbi,
	simulate_for_sbi,
)

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class SBISimulationBasedInference(CalibrationWorkflowBase):
	"""The SBI simulation-based inference method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		parameter_spec = self.specification.parameter_spec.parameters

		self.names = []
		self.parameters = []
		for spec in parameter_spec:
			name = spec.name
			self.names.append(name)

			data_type = spec.data_type
			if data_type == ParameterDataType.DISCRETE:
				lower_bound, upper_bound = self.get_parameter_bounds(spec)
				lower_bound = np.floor(lower_bound).astype("int")
				upper_bound = np.floor(upper_bound).astype("int")
				replicates = np.floor(upper_bound - lower_bound).astype("int")
				probabilities = torch.tensor([1 / replicates])
				probabilities = probabilities.repeat(replicates)
				base_distribution = dist.Categorical(probabilities)
				transforms = [
					dist.AffineTransform(
						loc=torch.Tensor([lower_bound]), scale=torch.Tensor([1])
					)
				]
				prior = dist.TransformedDistribution(base_distribution, transforms)
			else:
				distribution_name = (
					spec.distribution_name.replace("_", " ").title().replace(" ", "")
				)

				distribution_args = spec.distribution_args
				if distribution_args is None:
					distribution_args = []
				distribution_args = [torch.Tensor([arg]) for arg in distribution_args]

				distribution_kwargs = spec.distribution_kwargs
				if distribution_kwargs is None:
					distribution_kwargs = {}
				distribution_kwargs = {
					k: torch.Tensor([v]) for k, v in distribution_kwargs.items()
				}

				distribution_class = getattr(dist, distribution_name)

				prior = distribution_class(*distribution_args, **distribution_kwargs)

			self.parameters.append(prior)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		def simulator_func(theta: np.ndarray) -> np.ndarray:
			theta = theta.detach().cpu().numpy()
			parameters = {}
			for i, name in enumerate(self.names):
				parameters[name] = theta[i]

			sbi_kwargs = self.get_calibration_func_kwargs()

			observed_data = self.specification.observed_data
			simulation_id = get_simulation_uuid()
			results = self.calibration_func(
				parameters, simulation_id, observed_data, **sbi_kwargs
			)
			return results

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
		task, time_now, outdir = self.prepare_analyze()

		n_draws = self.specification.n_samples
		posterior_samples = self.posterior.sample(
			(n_draws,), x=self.specification.observed_data
		)

		for plot_func in [analysis.pairplot, analysis.marginal_plot]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=self.names)
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
				fig.savefig(outfile)
			else:
				fig.show()

		limits = []
		lower_limits, _ = posterior_samples.min(axis=0)
		upper_limits, _ = posterior_samples.max(axis=0)
		for i in range(len(self.names)):
			limits.append((lower_limits[i], upper_limits[i]))

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
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
				fig.savefig(outfile)
			else:
				fig.show()

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
			if outdir is not None:
				outfile = osp.join(
					outdir,
					f"{time_now}-{analysis.sbc_rank_plot.__name__}_{plot_type}.png",
				)
				fig.savefig(outfile)
			else:
				fig.show()

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
		outfile = osp.join(outdir, f"{time_now}-{task}_diagnostics.csv")
		check_stats_df.to_csv(outfile, index=False)

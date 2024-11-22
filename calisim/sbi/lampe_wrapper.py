"""Contains the implementations for simulation-based inference methods using
LAMPE

Implements the supported simulation-based inference methods using
the LAMPE library.

"""

from itertools import islice

import numpy as np
import torch
import torch.optim as optim
from lampe.data import JointLoader
from lampe.diagnostics import expected_coverage_mc
from lampe.inference import NPE, NPELoss
from lampe.plots import coverage_plot
from lampe.utils import GDStep
from matplotlib import pyplot as plt
from sbi import analysis as analysis

from ..base import SimulationBasedInferenceBase
from ..utils import PriorCollection


class LAMPESimulationBasedInference(SimulationBasedInferenceBase):
	"""The LAMPE simulation-based inference method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		super().specify()
		self.parameters: PriorCollection = PriorCollection(self.parameters)  # type: ignore[assignment]

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""

		sbi_kwargs = self.get_calibration_func_kwargs()

		def simulator_func(X: np.ndarray) -> np.ndarray:
			X = X.detach().cpu().numpy().T
			X = [X]
			results = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				sbi_kwargs,
			)
			results = results[0]
			return torch.from_numpy(results).float()

		loader = JointLoader(
			self.parameters, simulator_func, batch_size=1, vectorized=True
		)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		estimator = NPE(
			len(self.names),
			len(self.specification.observed_data),  # type: ignore[arg-type]
			**method_kwargs,
		)
		loss = NPELoss(estimator)
		optimizer = optim.Adam(estimator.parameters(), lr=self.specification.lr)
		step = GDStep(optimizer, clip=0.0)
		estimator.train()

		for epoch in range(self.specification.n_iterations):
			for theta, x in islice(loader, self.specification.num_simulations):
				neg_log_p = loss(theta, x)
				step(neg_log_p)
			if self.specification.verbose:
				print(f"Epoch {epoch + 1} : Negative log-likelihood {neg_log_p}")  # type: ignore[possibly-undefined]

		self.loader = loader
		self.estimator = estimator
		self.loss = loss
		self.optimizer = optimizer
		self.step = step

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()

		x_star = torch.from_numpy(self.specification.observed_data).float()
		n_draws = self.specification.n_samples
		with torch.no_grad():
			posterior_samples = self.estimator.flow(x_star).sample((n_draws,))

		for plot_func in [analysis.pairplot, analysis.marginal_plot]:
			plt.rcParams.update({"font.size": 8})
			fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=self.names)
			if outdir is not None:
				outfile = self.join(outdir, f"{time_now}-{plot_func.__name__}.png")
				fig.savefig(outfile)
			else:
				fig.show()

		n_simulations = self.specification.num_simulations
		levels, coverages = expected_coverage_mc(
			posterior=self.estimator.flow,
			pairs=((theta, x) for theta, x in islice(self.loader, n_simulations)),
		)

		fig = coverage_plot(levels, coverages, legend=task)
		if outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{coverage_plot.__name__}.png",
			)
			fig.savefig(outfile)
		else:
			fig.show()

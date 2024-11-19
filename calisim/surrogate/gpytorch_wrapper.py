"""Contains the implementations for surrogate modelling methods using GPyTorch

Implements the supported surrogate modelling methods using the GPyTorch library.

"""

import os.path as osp

import gpytorch
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from ..base import SurrogateBase
from ..utils import calibration_func_wrapper, extend_X


class SingleTaskGPRegressionModel(gpytorch.models.ExactGP):
	def __init__(
		self,
		train_x: torch.Tensor,
		train_y: torch.Tensor,
		likelihood: gpytorch.likelihoods.Likelihood,
	):
		super().__init__(train_x, train_y, likelihood)

		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, X: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		X = X - X.min(0)[0]
		X = 2 * (X / X.max(0)[0]) - 1
		mean_x = self.mean_module(X)
		covar_x = self.covar_module(X)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchSurrogateModel(SurrogateBase):
	"""The GPyTorch surrogate modelling method class."""

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		surrogate_kwargs = self.get_calibration_func_kwargs()
		n_samples = self.specification.n_samples

		X = self.specification.X
		if X is None:
			X = self.sample_parameters(n_samples)

		Y = self.specification.Y
		if Y is None:
			Y = calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				surrogate_kwargs,
			)

		self.Y_shape = Y.shape
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			X = extend_X(X, self.Y_shape[1])
			Y = Y.flatten()

		X_scaler = MinMaxScaler()
		X_scaler.fit(X)
		self.X_scaler = X_scaler

		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		X = torch.tensor(X_scaler.transform(X), dtype=torch.double, device=device)
		Y = torch.tensor(Y, dtype=torch.double, device=device)

		dataset = TensorDataset(X, Y)
		batch_size = self.specification.batch_size
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

		likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device).double()
		model = SingleTaskGPRegressionModel(X, Y, likelihood).to(device).double()

		optimizer = torch.optim.Adam(model.parameters(), lr=self.specification.lr)
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

		for epoch in range(self.specification.n_iterations):
			model.train()
			for X_batch, Y_batch in loader:
				optimizer.zero_grad()
				output = model(X_batch)
				loss = -mll(output, Y_batch)
				loss.backward()
				optimizer.step()

			if self.specification.verbose:
				print(
					f"Epoch: {epoch}",
					f"Training Loss: {loss.item()}",  # type: ignore[possibly-undefined]
				)

		self.emulator = model
		self.likelihood = likelihood
		self.optimizer = optimizer
		self.loader = loader
		self.device = device
		self.X = X
		self.Y = Y

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()

		model = self.emulator
		model.eval()
		likelihood = self.likelihood
		likelihood.eval()

		loader = self.loader
		X = []
		Y = []
		for X_batch, y_batch in loader:
			X.extend(X_batch)
			Y.extend(y_batch)
		X = torch.stack(X)
		Y = torch.stack(Y)
		X = X.detach().cpu().numpy()
		Y = Y.detach().cpu().numpy()

		n_samples = self.specification.n_samples
		X_sample = self.parameters.sample(n_samples, rule="sobol").T
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			X_sample = extend_X(X_sample, self.Y_shape[1])
		X_sample = torch.tensor(
			self.X_scaler.transform(X_sample), dtype=torch.double, device=self.device
		)

		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			sample_predictions = likelihood(model(X_sample))
			Y_sample = sample_predictions.mean.detach().cpu().numpy()
			sample_lower, sample_upper = sample_predictions.confidence_region()
			sample_lower, sample_upper = (
				sample_lower.detach().cpu().numpy(),
				sample_upper.detach().cpu().numpy(),
			)

		names = self.names.copy()
		if X.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(X, columns=names)

		output_label = self.specification.output_labels[0]  # type: ignore[index]
		if len(self.Y_shape) == 1:
			df[f"simulated_{output_label}"] = Y
			fig, axes = plt.subplots(
				nrows=len(self.names), figsize=self.specification.figsize
			)
			for i, parameter_name in enumerate(self.names):
				df.plot.scatter(
					parameter_name,
					f"simulated_{output_label}",
					ax=axes[i],
					title=f"simulated_{output_label} against {parameter_name}",
				)

			fig.tight_layout()
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{task}_plot_slice.png")
				fig.savefig(outfile)
			else:
				fig.show()

			fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)
			df = pd.DataFrame(
				{
					"index": np.arange(0, Y.shape[0], 1),
					"simulated": Y,
					"emulated": Y_sample,
				}
			)
			df.plot.scatter(
				"index", "simulated", ax=axes[0], title=f"Simulated {output_label}"
			)
			df.plot.scatter(
				"index", "emulated", ax=axes[1], title=f"Emulated {output_label}"
			)

			fig.tight_layout()
			if outdir is not None:
				outfile = osp.join(
					outdir, f"{time_now}-{task}_emulated_{output_label}.png"
				)
				fig.savefig(outfile)
			else:
				fig.show()
		else:
			if self.specification.flatten_Y:
				df[f"simulated_{output_label}"] = Y
				fig, axes = plt.subplots(
					nrows=len(self.names), figsize=self.specification.figsize
				)
				for i, parameter_name in enumerate(self.names):
					df.plot.scatter(
						parameter_name,
						f"simulated_{output_label}",
						ax=axes[i],
						title=f"simulated_{output_label} against {parameter_name}",
					)

				fig.tight_layout()
				if outdir is not None:
					outfile = osp.join(outdir, f"{time_now}-{task}_plot_slice.png")
					fig.savefig(outfile)
				else:
					fig.show()

				reshaped_Y_sample = Y_sample.reshape(self.Y_shape)
				Y = self.Y.reshape(self.Y_shape).detach().cpu().numpy()
				indx = np.arange(0, reshaped_Y_sample.shape[1], 1)

				fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)

				for i in range(Y.shape[0]):
					axes[0].plot(indx, Y[i])
				axes[0].set_title(f"Simulated {output_label}")

				for i in range(reshaped_Y_sample.shape[0]):
					axes[1].plot(indx, reshaped_Y_sample[i])
				axes[1].set_title(f"Emulated {output_label}")

				fig.tight_layout()
				if outdir is not None:
					outfile = osp.join(
						outdir, f"{time_now}-{task}_emulated_{output_label}.png"
					)
					fig.savefig(outfile)
				else:
					fig.show()

		if outdir is None:
			return

		metrics = []
		for metric_func in [
			gpytorch.metrics.mean_standardized_log_loss,
			gpytorch.metrics.mean_squared_error,
			gpytorch.metrics.mean_absolute_error,
			gpytorch.metrics.quantile_coverage_error,
			gpytorch.metrics.negative_log_predictive_density,
		]:
			metric_name = metric_func.__name__
			metric_score = (
				metric_func(
					sample_predictions,
					torch.tensor(Y_sample, dtype=torch.double, device=self.device),
				)
				.cpu()
				.detach()
				.numpy()
				.item()
			)
			metrics.append({"metric_name": metric_name, "metric_score": metric_score})

		metric_df = pd.DataFrame(metrics)
		outfile = osp.join(outdir, f"{time_now}_{task}_metrics.csv")
		metric_df.to_csv(outfile, index=False)

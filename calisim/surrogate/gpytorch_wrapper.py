"""Contains the implementations for surrogate modelling methods using GPyTorch

Implements the supported surrogate modelling methods using the GPyTorch library.

"""

import os.path as osp

import chaospy
import gpytorch
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class SingleTaskGPRegressionModel(gpytorch.models.ExactGP):
	def __init__(
		self,
		train_x: torch.Tensor,
		train_y: torch.Tensor,
		likelihood: gpytorch.likelihoods.Likelihood,
	):
		super().__init__(train_x, train_y, likelihood)

		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.GridInterpolationKernel(
			gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
			grid_size=100,
			num_dims=train_x.size(-1),
		)

	def forward(self, X: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		X = X - X.min(0)[0]
		X = 2 * (X / X.max(0)[0]) - 1
		mean_x = self.mean_module(X)
		covar_x = self.covar_module(X)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchSurrogateModel(CalibrationWorkflowBase):
	"""The GPyTorch surrogate modelling method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		self.names = []
		self.data_types = []
		parameters = []

		parameter_spec = self.specification.parameter_spec.parameters
		for spec in parameter_spec:
			parameter_name = spec.name
			self.names.append(parameter_name)

			data_type = spec.data_type
			self.data_types.append(data_type)

			distribution_name = (
				spec.distribution_name.replace("_", " ").title().replace(" ", "")
			)
			distribution_args = spec.distribution_args
			if distribution_args is None:
				distribution_args = []

			distribution_kwargs = spec.distribution_kwargs
			if distribution_kwargs is None:
				distribution_kwargs = {}

			dist_instance = getattr(chaospy, distribution_name)
			parameter = dist_instance(*distribution_args, **distribution_kwargs)
			parameters.append(parameter)

		self.parameters = chaospy.J(*parameters)

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		n_samples = self.specification.n_samples
		X = self.parameters.sample(n_samples, rule="sobol").T

		def surrogate_func(
			X: np.ndarray,
			observed_data: pd.DataFrame | np.ndarray,
			parameter_names: list[str],
			data_types: list[ParameterDataType],
			surrogate_kwargs: dict,
		) -> np.ndarray:
			parameters = []
			for theta in X:
				parameter_set = {}
				for i, parameter_value in enumerate(theta):
					parameter_name = parameter_names[i]
					data_type = data_types[i]
					if data_type == ParameterDataType.CONTINUOUS:
						parameter_set[parameter_name] = parameter_value
					else:
						parameter_set[parameter_name] = int(parameter_value)
				parameters.append(parameter_set)

			simulation_ids = [get_simulation_uuid() for _ in range(len(parameters))]

			if self.specification.batched:
				results = self.calibration_func(
					parameters, simulation_ids, observed_data, **surrogate_kwargs
				)
			else:
				results = []
				for i, parameter in enumerate(parameters):
					simulation_id = simulation_ids[i]
					result = self.calibration_func(
						parameter,
						simulation_id,
						observed_data,
						**surrogate_kwargs,
					)
					results.append(result)

			results = np.array(results)
			return results

		surrogate_kwargs = self.specification.calibration_func_kwargs
		if surrogate_kwargs is None:
			surrogate_kwargs = {}

		Y = surrogate_func(
			X,
			self.specification.observed_data,
			self.names,
			self.data_types,
			surrogate_kwargs,
		)

		self.Y_shape = Y.shape
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			design_list = []
			for i in range(X.shape[0]):
				for j in range(self.Y_shape[1]):
					row = np.append(X[i], j)
					design_list.append(row)
			X = np.array(design_list)
			Y = Y.flatten()

		X_scaler = MinMaxScaler()
		X_scaler.fit(X)
		self.X_scaler = X_scaler

		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		X = torch.tensor(X_scaler.transform(X), device=device)
		Y = torch.tensor(Y, device=device)

		dataset = TensorDataset(X, Y)
		batch_size = self.specification.batch_size
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

		likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
		model = SingleTaskGPRegressionModel(X, Y, likelihood).to(device)

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
			design_list = []
			for i in range(X_sample.shape[0]):
				for j in range(self.Y_shape[1]):
					row = np.append(X_sample[i], j)
					design_list.append(row)
			X_sample = np.array(design_list)
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
				print(Y.shape)
				return
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

				Y_sample = Y_sample.reshape(self.Y_shape)
				Y = self.Y.reshape(self.Y_shape)
				indx = np.arange(0, Y_sample.shape[1], 1)

				fig, axes = plt.subplots(nrows=2, figsize=self.specification.figsize)

				for i in range(Y.shape[0]):
					axes[0].plot(indx, Y[i])
				axes[0].set_title(f"Simulated {output_label}")

				for i in range(Y_sample.shape[0]):
					axes[1].plot(indx, Y_sample[i])
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

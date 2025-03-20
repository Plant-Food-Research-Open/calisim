"""Contains the implementations for surrogate modelling methods using GPyTorch

Implements the supported surrogate modelling methods using the GPyTorch library.

"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from ..base import SurrogateBase
from ..estimators import get_single_task_exact_gp


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
			Y = self.calibration_func_wrapper(
				X,
				self,
				self.specification.observed_data,
				self.names,
				self.data_types,
				surrogate_kwargs,
			)

		self.Y_shape = Y.shape
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			X = self.extend_X(X, self.Y_shape[1])
			Y = Y.flatten()

		X_scaler = MinMaxScaler()
		X_scaler.fit(X)
		self.X_scaler = X_scaler

		if torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"

		X = torch.tensor(X_scaler.transform(X), dtype=torch.double)
		Y = torch.tensor(Y, dtype=torch.double)

		model = get_single_task_exact_gp(
			lr=self.specification.lr,
			max_epochs=self.specification.n_iterations,
			device=device,
		)
		model.fit(X, Y)

		self.emulator = model
		self.device = device
		self.X = X
		self.Y = Y

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()

		model = self.emulator
		X = self.X
		Y = self.Y

		n_samples = self.specification.n_samples
		X_sample = self.parameters.sample(n_samples, rule="sobol").T
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			X_sample = self.extend_X(X_sample, self.Y_shape[1])
		X_sample = torch.tensor(self.X_scaler.transform(X_sample), dtype=torch.double)
		Y_sample = model.predict(X_sample, return_std=False)

		names = self.names.copy()
		if X.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(X, columns=names)

		if self.specification.use_shap and outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-param_importances.png",
			)
			self.calculate_shap_importances(
				df, self.emulator, names, self.specification.test_size, outfile
			)

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
			self.present_fig(fig, outdir, time_now, task, experiment_name, "plot_slice")

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
			self.present_fig(
				fig, outdir, time_now, task, experiment_name, f"emulated-{output_label}"
			)
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
				self.present_fig(
					fig, outdir, time_now, task, experiment_name, "plot_slice"
				)

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
				self.present_fig(
					fig,
					outdir,
					time_now,
					task,
					experiment_name,
					f"emulated-{output_label}",
				)

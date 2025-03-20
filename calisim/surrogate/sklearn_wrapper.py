"""Contains the implementations for surrogate modelling methods using Scikit-Learn

Implements the supported surrogate modelling methods using the Scikit-Learn library.

"""

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gp
import sklearn.kernel_ridge as kernel_ridge
import sklearn.linear_model as lm
import sklearn.neighbors as neighbors
import sklearn.svm as svm
from matplotlib import pyplot as plt

from ..base import SurrogateBase
from ..estimators import EmukitEstimator, FunctionalChaosEstimator, KrigingEstimator


class SklearnSurrogateModel(SurrogateBase):
	"""The Scikit-Learn surrogate modelling method class."""

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

		emulator_name = self.specification.method
		emulators = dict(
			gp=gp.GaussianProcessRegressor,
			emukit_gp=EmukitEstimator,
			openturns_functional_chaos=FunctionalChaosEstimator,
			openturns_kriging=KrigingEstimator,
			rf=ensemble.RandomForestRegressor,
			gb=ensemble.GradientBoostingRegressor,
			lr=lm.LinearRegression,
			elastic=lm.MultiTaskElasticNet,
			ridge=lm.Ridge,
			knn=neighbors.KNeighborsRegressor,
			kr=kernel_ridge.KernelRidge,
			linear_svm=svm.LinearSVR,
			nu_svm=svm.NuSVR,
		)
		emulator_class = emulators.get(emulator_name, None)
		if emulator_class is None:
			raise ValueError(
				f"Unsupported emulator: {emulator_name}.",
				f"Supported emulators are {', '.join(emulators)}",
			)

		method_kwargs = self.specification.method_kwargs
		if method_kwargs is None:
			method_kwargs = {}

		self.Y_shape = Y.shape
		if (
			self.specification.flatten_Y
			and len(self.Y_shape) > 1
			and self.specification.X is None
		):
			X = self.extend_X(X, self.Y_shape[1])
			Y = Y.flatten()

		self.emulator = emulator_class(**method_kwargs)
		self.emulator.fit(X, Y)

		self.X = X
		self.Y = Y

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, experiment_name, outdir = self.prepare_analyze()
		output_label = self.specification.output_labels[0]  # type: ignore[index]

		names = self.names.copy()
		if self.X.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(self.X, columns=names)

		n_samples = self.specification.n_samples
		X_sample = self.sample_parameters(n_samples)
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			X_sample = self.extend_X(X_sample, self.Y_shape[1])
		Y_sample = self.emulator.predict(X_sample)

		if self.specification.use_shap and outdir is not None:
			outfile = self.join(
				outdir,
				f"{time_now}-{task}-{experiment_name}-param_importances.png",
			)
			self.calculate_shap_importances(
				X_sample, self.emulator, names, self.specification.test_size, outfile
			)

		if len(self.Y_shape) == 1:
			df[f"simulated_{output_label}"] = self.Y
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
					"index": np.arange(0, self.Y.shape[0], 1),
					"simulated": self.Y,
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
				df[f"simulated_{output_label}"] = self.Y
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
				self.present_fig(
					fig,
					outdir,
					time_now,
					task,
					experiment_name,
					f"emulated-{output_label}",
				)
			else:
				output_labels = self.specification.output_labels
				if len(output_labels) != self.Y_shape[-1]:  # type: ignore[arg-type]
					output_labels = [f"Output_{y}" for y in range(self.Y_shape[-1])]

				fig, axes = plt.subplots(
					nrows=len(self.names) * len(output_labels),  # type: ignore[arg-type]
					ncols=2,
					figsize=self.specification.figsize,
				)

				row_indx = 0
				for x_indx, parameter_name in enumerate(self.names):
					for y_indx, output_label in enumerate(output_labels):  # type: ignore[arg-type]
						axes[row_indx, 0].scatter(self.X[:, x_indx], self.Y[:, y_indx])
						axes[row_indx, 0].set_xlabel(parameter_name)
						axes[row_indx, 0].set_ylabel(output_label)
						axes[row_indx, 0].set_title(f"Simulated {output_label}")

						axes[row_indx, 1].scatter(
							X_sample[:, x_indx], Y_sample[:, y_indx]
						)
						axes[row_indx, 1].set_xlabel(parameter_name)
						axes[row_indx, 1].set_ylabel(output_label)
						axes[row_indx, 1].set_title(f"Emulated {output_label}")

						row_indx += 1

				self.present_fig(
					fig, outdir, time_now, task, experiment_name, "plot_slice"
				)

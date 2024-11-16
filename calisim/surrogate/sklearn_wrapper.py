"""Contains the implementations for surrogate modelling methods using Scikit-Learn

Implements the supported surrogate modelling methods using the Scikit-Learn library.

"""

import os.path as osp

import chaospy
import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gp
import sklearn.kernel_ridge as kernel_ridge
import sklearn.linear_model as lm
import sklearn.neighbors as neighbors
import sklearn.svm as svm
from matplotlib import pyplot as plt

from ..base import CalibrationWorkflowBase
from ..data_model import ParameterDataType
from ..utils import get_simulation_uuid


class SklearnSurrogateModel(CalibrationWorkflowBase):
	"""The Scikit-Learn surrogate modelling method class."""

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

		surrogate_kwargs = self.get_calibration_func_kwargs()

		Y = surrogate_func(
			X,
			self.specification.observed_data,
			self.names,
			self.data_types,
			surrogate_kwargs,
		)

		emulator_name = self.specification.method
		emulators = dict(
			gp=gp.GaussianProcessRegressor,
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
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			design_list = []
			for i in range(X.shape[0]):
				for j in range(self.Y_shape[1]):
					row = np.append(X[i], j)
					design_list.append(row)
			X = np.array(design_list)
			Y = Y.flatten()

		self.emulator = emulator_class(**method_kwargs)
		self.emulator.fit(X, Y)

		self.X = X
		self.Y = Y

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		task, time_now, outdir = self.prepare_analyze()
		output_label = self.specification.output_labels[0]  # type: ignore[index]

		names = self.names.copy()
		if self.X.shape[1] > len(names):
			names.append("_dummy_index")
		df = pd.DataFrame(self.X, columns=names)

		n_samples = self.specification.n_samples
		X_sample = self.parameters.sample(n_samples, rule="sobol").T
		if self.specification.flatten_Y and len(self.Y_shape) > 1:
			design_list = []
			for i in range(X_sample.shape[0]):
				for j in range(self.Y_shape[1]):
					row = np.append(X_sample[i], j)
					design_list.append(row)
			X_sample = np.array(design_list)
		Y_sample = self.emulator.predict(X_sample, return_std=False)

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

			fig.tight_layout()
			if outdir is not None:
				outfile = osp.join(outdir, f"{time_now}-{task}_plot_slice.png")
				fig.savefig(outfile)
			else:
				fig.show()

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

				fig.tight_layout()
				if outdir is not None:
					outfile = osp.join(outdir, f"{time_now}-{task}_plot_slice.png")
					fig.savefig(outfile)
				else:
					fig.show()

"""Contains the implementations for optimisation methods using Optuna

Implements the supported optimisation methods using the Optuna library.

"""

from ..base import CalibrationWorkflowBase


class OptunaOptimisation(CalibrationWorkflowBase):
	"""The Optuna method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("specify() method not implemented.")

	def execute(self) -> None:
		"""Execute the simulation calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("execute() method not implemented.")

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure.

		Raises:
		    NotImplementedError:
		        Error raised for the unimplemented abstract method.
		"""
		raise NotImplementedError("analyze() method not implemented.")

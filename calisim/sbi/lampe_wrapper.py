"""Contains the implementations for simulation-based inference methods using
LAMPE

Implements the supported simulation-based inference methods using
the LAMPE library.

"""

from ..base import CalibrationWorkflowBase


class LAMPESimulationBasedInference(CalibrationWorkflowBase):
	"""The LAMPE simulation-based inference method class."""

	def specify(self) -> None:
		"""Specify the parameters of the model calibration procedure."""
		pass
		# parameter_spec = self.specification.parameter_spec

	def execute(self) -> None:
		"""Execute the simulation calibration procedure."""
		pass
		# output_labels = self.specification.output_labels

	def analyze(self) -> None:
		"""Analyze the results of the simulation calibration procedure."""
		pass
		# task, time_now, outdir = self.prepare_analyze()

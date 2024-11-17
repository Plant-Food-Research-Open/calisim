"""Contains the implementations for the simulation-based inference methods

Implements the supported simulation-based inference methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel
from .lampe_wrapper import LAMPESimulationBasedInference
from .sbi_wrapper import SBISimulationBasedInference

TASK = "simulation_based_inference"
IMPLEMENTATIONS: dict[str, type[CalibrationWorkflowBase]] = dict(
	lampe=LAMPESimulationBasedInference, sbi=SBISimulationBasedInference
)


def get_implementations() -> dict[str, type[CalibrationWorkflowBase]]:
	"""Get the calibration implementations for simulation-based inference.

	Returns:
		Dict[str, type[CalibrationWorkflowBase]]: The dictionary of
			calibration implementations for simulation-based inference.
	"""
	return IMPLEMENTATIONS


class SimulationBasedInferenceMethodModel(CalibrationModel):
	"""The simulation-based inference method data model.

	Args:
	    BaseModel (CalibrationModel): The calibration base model class.
	"""

	num_simulations: int = Field(
		description="The number of simulations to run", default=25
	)


class SimulationBasedInferenceMethod(CalibrationMethodBase):
	"""The simulation-based inference method class."""

	def __init__(
		self,
		calibration_func: Callable,
		specification: SimulationBasedInferenceMethodModel,
		engine: str = "sbi",
		implementation: CalibrationWorkflowBase | None = None,
	) -> None:
		"""SimulationBasedInferenceMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (SimulationBasedInferenceMethodModel): The
				calibration specification.
		    engine (str, optional): The simulation-based inference
				backend. Defaults to "sbi".
			implementation (CalibrationWorkflowBase | None): The
				calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			IMPLEMENTATIONS,
			implementation,
		)

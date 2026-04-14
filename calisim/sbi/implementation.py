"""Contains the implementations for the simulation-based inference methods

Implements the supported simulation-based inference methods.

"""

from collections.abc import Callable

from pydantic import Field

from ..base import CalibrationMethodBase, CalibrationWorkflowBase
from ..data_model import CalibrationModel

TASK = "sbi"
BASE_IMPLEMENTATIONS: dict[str, str] = dict(
	lampe=f"calisim.{TASK}.lampe_wrapper:LAMPESimulationBasedInference",
	sbi=f"calisim.{TASK}.sbi_wrapper:SBISimulationBasedInference",
)


def get_implementations() -> dict[str, str]:
	"""Get the calibration implementations for simulation-based inference.

	Returns:
		Dict[str, str]: The dictionary of
			calibration implementations for simulation-based inference.
	"""
	implementations = dict(BASE_IMPLEMENTATIONS)
	implementations.update(CalibrationMethodBase.load_external_implementations(TASK))
	return implementations


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
		implementation: type[CalibrationWorkflowBase] | None = None,
	) -> None:
		"""SimulationBasedInferenceMethod constructor.

		Args:
			calibration_func (Callable): The calibration function.
				For example, a simulation function or objective function.
		    specification (SimulationBasedInferenceMethodModel): The
				calibration specification.
		    engine (str, optional): The simulation-based inference
				backend. Defaults to "sbi".
			implementation (type[CalibrationWorkflowBase] | None): The
				calibration workflow implementation.
		"""
		super().__init__(
			calibration_func,
			specification,
			TASK,
			engine,
			get_implementations(),
			implementation,
		)

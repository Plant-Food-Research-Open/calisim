"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for various
simulation calibration procedures.

"""

from typing import Any

from pydantic import BaseModel


class Config:
	arbitrary_types_allowed = True


class ParameterIntervalModel(BaseModel):
	"""The parameter interval data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	name: str
	lower_bound: float
	upper_bound: float
	data_type: str


class DistributionModel(BaseModel):
	"""The probability distribution data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	name: str
	dist_name: str
	dist_kwargs: dict[str, Any]


class CalibrationMethodModel(BaseModel):
	"""The calibration method data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	parameter_spec: list[ParameterIntervalModel] | list[DistributionModel]
	calibration_spec: dict[str, Any]
	analysis_spec: dict[str, Any]

"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for various
simulation calibration procedures.

"""

from enum import Enum
from typing import Any

from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
	"""Base Pydantic data model.

	Args:
	    PydanticBaseModel (PydanticBaseModel):
	        The Pydantic Base model class.
	"""

	class Config:
		arbitrary_types_allowed = True


class ParameterDataType(Enum):
	DISCRETE = "discrete"
	CONTINUOUS = "continuous"


class ParameterIntervalModel(BaseModel):
	"""The parameter interval data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	name: str
	lower_bound: float
	upper_bound: float
	data_type: ParameterDataType


class DistributionModel(BaseModel):
	"""The probability distribution data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	name: str
	dist_name: str
	dist_kwargs: dict[str, Any]


class CalibrationModel(BaseModel):
	"""The calibration data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	experiment_name: str | None = "default"
	outdir: str | None = None
	verbose: bool | None = False


class IntervalCalibrationModel(CalibrationModel):
	"""The interval-based calibration data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	parameter_spec: list[ParameterIntervalModel]


class DistributionCalibrationModel(CalibrationModel):
	"""The distribution-based calibration data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	parameter_spec: list[DistributionModel]

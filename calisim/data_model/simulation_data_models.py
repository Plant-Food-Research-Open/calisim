"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for various
simulation calibration procedures.

"""

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
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


class DistributionModel(BaseModel):
	"""The probability distribution data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	name: str
	distribution_name: str = "uniform"
	distribution_args: list | None = None
	distribution_kwargs: dict[str, Any] | None = None
	data_type: ParameterDataType = ParameterDataType.CONTINUOUS


class CalibrationModel(BaseModel):
	"""The calibration data model.

	Args:
	    BaseModel (BaseModel):
	        The Pydantic Base model class.
	"""

	parameter_spec: list[DistributionModel]
	experiment_name: str | None = "default"
	outdir: str | None = None
	method: str = ""
	method_kwargs: dict[str, Any] | None = None
	calibration_kwargs: dict[str, Any] | None = None
	analyze_kwargs: dict[str, Any] | None = None
	observed_data: np.ndarray | pd.DataFrame | None = None
	n_samples: int = 1
	n_init: int = 1
	n_iterations: int = 1
	random_seed: int | None = None
	n_jobs: int = 1
	walltime: int = 1
	output_labels: list[str] | None = None
	groups: list[str] | None = None
	parallel_backend: str = ""
	vectorize: bool = False
	verbose: bool = False
	figsize: tuple[int, int] = (12, 12)

"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for various
simulation calibration procedures.

"""

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class BaseModel(PydanticBaseModel):
	"""Base Pydantic data model.

	Args:
	    PydanticBaseModel (PydanticBaseModel): The Pydantic Base model class.
	"""

	class Config:
		arbitrary_types_allowed = True


class ParameterDataType(Enum):
	DISCRETE: str = "discrete"
	CONTINUOUS: str = "continuous"


class DistributionModel(BaseModel):
	"""The probability distribution data model.

	Args:
	    BaseModel (BaseModel): The Pydantic Base model class.
	"""

	name: str = Field(description="The parameter name")
	distribution_name: str = Field(
		description="The distribution name", default="uniform"
	)
	distribution_args: list | None = Field(
		description="The distribution arguments", default=None
	)
	distribution_kwargs: dict[str, Any] | None = Field(
		description="The distribution named arguments", default=None
	)
	data_type: ParameterDataType = Field(
		description="The distribution data type", default=ParameterDataType.CONTINUOUS
	)


class ParameterSpecification(BaseModel):
	"""The collection of parameters.

	Args:
	    BaseModel (BaseModel): The Pydantic Base model class.
	"""

	parameters: list[DistributionModel] | None = Field(
		description="The parameter specification list", default=None
	)


class CalibrationModel(BaseModel):
	"""The calibration data model.

	Args:
	    BaseModel (BaseModel): The Pydantic Base model class.
	"""

	parameter_spec: ParameterSpecification | None = Field(
		description="The parameter specification", default=None
	)
	experiment_name: str | None = Field(
		description="The modelling experiment name", default="default"
	)
	outdir: str | None = Field(
		description="The output directory for modelling results", default=None
	)
	method: str = Field(description="The calibration method or algorithm", default="")
	method_kwargs: dict[str, Any] | None = Field(
		description="The calibration method named arguments", default=None
	)
	calibration_func_kwargs: dict[str, Any] | None = Field(
		description="The calibration function named arguments", default=None
	)
	pass_calibration_workflow: bool | str | None = Field(
		description="Pass the calibration workflow into the calibration function",
		default=None,
	)
	analyze_kwargs: dict[str, Any] | None = Field(
		description="The analyze step named arguments", default=None
	)
	X: np.ndarray | pd.DataFrame | list | None = Field(
		description="The simulation input data", default=None
	)
	Y: np.ndarray | pd.DataFrame | list | None = Field(
		description="The simulation output data", default=None
	)
	observed_data: np.ndarray | pd.DataFrame | None = Field(
		description="The empirical or observed data", default=None
	)
	n_samples: int = Field(description="The number of samples to take", default=1)
	n_chains: int = Field(description="The number of Markov chains", default=1)
	n_init: int = Field(description="The number of initial samples or steps", default=1)
	n_iterations: int = Field(
		description="The number of iterations for sequential calibrators", default=1
	)
	lr: float = Field(
		description="The learning rate of the model",
		default=0.01,
	)
	random_seed: int | None = Field(
		description="The random seed for replicability", default=None
	)
	n_jobs: int = Field(description="The number of jobs to run in parallel", default=1)
	walltime: int = Field(description="The maximum calibration walltime", default=1)
	output_labels: list[str] | None = Field(
		description="The list of simulation output names", default=None
	)
	groups: list[str] | None = Field(
		description="The list of parameter groups", default=None
	)
	parallel_backend: str = Field(
		description="The backend engine to run parallel jobs", default=""
	)
	batched: bool = Field(description="Whether to batch the simulations", default=False)
	verbose: bool = Field(
		description="Whether to print calibration messages", default=False
	)
	figsize: tuple[int, int] = Field(
		description="The figure size for visualisations", default=(12, 12)
	)

import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.uncertainty import (
	UncertaintyAnalysisMethod,
	UncertaintyAnalysisMethodModel,
)
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = ParameterSpecification(
	parameters=[
		DistributionModel(
			name="alpha",
			distribution_name="uniform",
			distribution_args=[0.45, 0.55],
			data_type=ParameterDataType.CONTINUOUS,
		),
		DistributionModel(
			name="beta",
			distribution_name="uniform",
			distribution_args=[0.02, 0.03],
			data_type=ParameterDataType.CONTINUOUS,
		),
	]
)


def uncertainty_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


outdir = get_examples_outdir()
specification = UncertaintyAnalysisMethodModel(
	experiment_name="openturns_uncertainty_analysis",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	solver="kriging",
	method="linear",
	algorithm="least_squares",
	order=4,
	n_samples=100,
	n_out=len(observed_data),
	output_labels=["Lynx"],
	batch=False,
	calibration_func_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(rule="cholesky", normed=False, cross_truncation=1.0),
)

calibrator = UncertaintyAnalysisMethod(
	calibration_func=uncertainty_func, specification=specification, engine="openturns"
)

calibrator.specify().execute().analyze()

print(f"Results written to: {outdir}")

import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.example_models import LotkaVolterraModel
from calisim.quadrature import (
	QuadratureMethod,
	QuadratureMethodModel,
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


def quadrature_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	return simulated_data


outdir = get_examples_outdir()
specification = QuadratureMethodModel(
	experiment_name="emukit_quadrature",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_init=5,
	n_iterations=10,
	n_samples=50,
	kernel="QuadratureRBFLebesgueMeasure",
	measure="LebesgueMeasure",
	method_kwargs=dict(noise_var=1e-4),
	calibration_func_kwargs=dict(t=observed_data.year),
	batched=False,
)


calibrator = QuadratureMethod(
	calibration_func=quadrature_func,
	specification=specification,
	engine="emukit",
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")

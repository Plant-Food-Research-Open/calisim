import numpy as np
import pandas as pd

from calisim.data_model import DistributionModel, ParameterDataType
from calisim.example_models import LotkaVolterraModel
from calisim.optimisation import OptimisationMethod, OptimisationMethodModel
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	DistributionModel(
		name="alpha",
		dist_name="uniform",
		dist_args=[0.45, 0.55],
		data_type=ParameterDataType.CONTINUOUS,
	),
	DistributionModel(
		name="beta",
		dist_name="uniform",
		dist_args=[0.02, 0.03],
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def objective(
	parameters: dict, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy


specification = OptimisationMethodModel(
	experiment_name="botorch_optimisation",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=get_examples_outdir(),
	directions=["minimize"],
	n_init=5,
	n_samples=20,
	verbose=True,
	calibration_kwargs=dict(t=observed_data.year),
)

calibrator = OptimisationMethod(
	calibration_func=objective, specification=specification, engine="botorch"
)

calibrator.specify().execute().analyze()

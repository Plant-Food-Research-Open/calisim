import pandas as pd

from calisim.data_model import ParameterDataType, ParameterIntervalModel
from calisim.example_models import LotkaVolterraModel
from calisim.optimisation import OptimisationMethod, OptimisationMethodModel
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	ParameterIntervalModel(
		name="alpha",
		lower_bound=0.45,
		upper_bound=0.55,
		data_type=ParameterDataType.CONTINUOUS,
	),
	ParameterIntervalModel(
		name="beta",
		lower_bound=0.02,
		upper_bound=0.03,
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def objective(
	parameters: dict, observed_data: pd.DataFrame | None
) -> float | list[float]:
	simulation_parameters = dict(
		h0=34.0, l0=5.9, t=observed_data.year, gamma=0.84, delta=0.026
	)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters)
	observed_data = observed_data.drop(columns=["year"])

	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy


specification = OptimisationMethodModel(
	experiment_name="optuna_optimisation",
	parameter_spec=parameter_spec,
	observed_data=observed_data,
	outdir=get_examples_outdir(),
	sampler="tpes",
	directions=["minimize"],
	n_samples=20,
	sampler_kwargs=dict(n_startup_trials=10),
)

calibrator = OptimisationMethod(
	calibration_func=objective, specification=specification, engine="optuna"
)

calibrator.specify().execute().analyze()

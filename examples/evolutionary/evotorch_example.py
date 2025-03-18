import numpy as np
import pandas as pd

from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
)
from calisim.evolutionary import (
	EvolutionaryMethod,
	EvolutionaryMethodModel,
)
from calisim.example_models import LotkaVolterraModel
from calisim.statistics import MeanSquaredError
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


def evolutionary_func(
	parameters: dict, simulation_id: str, observed_data: np.ndarray | None, t: pd.Series
) -> float | list[float]:
	simulation_parameters = dict(h0=34.0, l0=5.9, t=t, gamma=0.84, delta=0.026)

	for k in ["alpha", "beta"]:
		simulation_parameters[k] = parameters[k]

	simulated_data = model.simulate(simulation_parameters).lynx.values
	metric = MeanSquaredError()
	discrepancy = metric.calculate(observed_data, simulated_data)
	return discrepancy


outdir = get_examples_outdir()
specification = EvolutionaryMethodModel(
	experiment_name="evotorch_evolutionary",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_iterations=10,
	n_samples=10,
	method="ga",
	directions=["minimize"],
	operators=dict(
		OnePointCrossOver=dict(tournament_size=4),
		GaussianMutation=dict(stdev=0.1),
		PolynomialMutation=dict(eta=20.0),
	),
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	calibration_func_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(popsize=100, elitist=True, re_evaluate=True),
)

calibrator = EvolutionaryMethod(
	calibration_func=evolutionary_func, specification=specification, engine="evotorch"
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")

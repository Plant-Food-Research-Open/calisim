import numpy as np
import pandas as pd

from calisim.abc import (
	ApproximateBayesianComputationMethod,
	ApproximateBayesianComputationMethodModel,
)
from calisim.data_model import (
	DistributionModel,
	ParameterDataType,
	ParameterSpecification,
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
			distribution_name="normal",
			distribution_args=[0.5, 0.03],
			data_type=ParameterDataType.CONTINUOUS,
		),
		DistributionModel(
			name="beta",
			distribution_name="normal",
			distribution_args=[0.025, 0.003],
			data_type=ParameterDataType.CONTINUOUS,
		),
	]
)


def abc_func(
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
specification = ApproximateBayesianComputationMethodModel(
	experiment_name="elfi_approximate_bayesian_computation",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_samples=100,
	walltime=3,  # minutes
	n_iterations=5,
	method="adaptive_threshold_smc",
	quantile=0.2,
	output_labels=["Lynx"],
	verbose=True,
	batched=False,
	calibration_func_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(
		initial_quantile=0.20,
		q_threshold=0.99,
	),
)

calibrator = ApproximateBayesianComputationMethod(
	calibration_func=abc_func, specification=specification, engine="elfi"
)

calibrator.specify().execute().analyze()

result_artifacts = "\n".join(calibrator.get_artifacts())
print(f"View results: \n{result_artifacts}")

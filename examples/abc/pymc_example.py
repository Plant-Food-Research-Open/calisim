import numpy as np
import pandas as pd

from calisim.abc import (
	ApproximateBayesianComputationMethod,
	ApproximateBayesianComputationMethodModel,
)
from calisim.data_model import DistributionModel, ParameterDataType
from calisim.example_models import LotkaVolterraModel
from calisim.statistics import MeanSquaredError
from calisim.utils import get_examples_outdir

model = LotkaVolterraModel()
observed_data = model.get_observed_data()

parameter_spec = [
	DistributionModel(
		name="alpha",
		distribution_name="normal",
		distribution_args=[0.4, 0.03],
		data_type=ParameterDataType.CONTINUOUS,
	),
	DistributionModel(
		name="beta",
		distribution_name="normal",
		distribution_args=[0.025, 0.003],
		data_type=ParameterDataType.CONTINUOUS,
	),
]


def abc_func(
	parameters: dict, observed_data: np.ndarray | None, t: pd.Series
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
	experiment_name="pymc_approximate_bayesian_computation",
	parameter_spec=parameter_spec,
	observed_data=observed_data.lynx.values,
	outdir=outdir,
	n_samples=3,
	n_chains=3,
	n_jobs=1,
	epsilon=0.1,
	sum_stat="identity",
	output_labels=["Lynx"],
	verbose=True,
	vectorize=False,
	calibration_kwargs=dict(t=observed_data.year),
	method_kwargs=dict(compute_convergence_checks=True, return_inferencedata=True),
)

calibrator = ApproximateBayesianComputationMethod(
	calibration_func=abc_func, specification=specification, engine="pymc"
)

calibrator.specify().execute().analyze()

print(f"Results written to: {outdir}")

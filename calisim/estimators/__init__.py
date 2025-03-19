import importlib

from .emukit_estimator import EmukitEstimator
from .openturns_estimator import FunctionalChaosEstimator, KrigingEstimator

__all__ = [
	EmukitEstimator,
	FunctionalChaosEstimator,
	KrigingEstimator,
]

if importlib.util.find_spec("gpytorch") is not None:
	from .gpytorch_estimator import (
		SingleTaskGPRegressionModel,
		get_single_task_exact_gp,
	)

	__all__.extend([SingleTaskGPRegressionModel, get_single_task_exact_gp])  # type: ignore[list-item]

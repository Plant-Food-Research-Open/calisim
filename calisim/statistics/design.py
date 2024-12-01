"""Contains utilities for the design of experiments.

This module defines utility functions for various
experimental design methods.
"""

from itertools import product

import numpy as np

from ..data_model import ParameterSpecification


def get_full_factorial_design(parameter_spec: ParameterSpecification) -> np.ndarray:
	"""Get a full factorial design from a parameter specification.

	Args:
		parameter_spec (ParameterSpecification): The parameter specification.

	Returns:
		np.ndarray: The full factorial design.
	"""
	factorial_design = []
	for parameter in parameter_spec.parameters:
		parameter_values = parameter.parameter_values
		if parameter_values is not None:
			factorial_design.append(parameter_values)

	cartesian_product = list(product(*factorial_design))
	cartesian_product = [np.array(combination) for combination in cartesian_product]
	return np.array(cartesian_product)

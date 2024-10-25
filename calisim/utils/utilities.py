"""Contains utility functions for calibration

This module defines various utility functions
for the calibration of simulations.

"""

import os.path as osp
import uuid
from datetime import datetime


def get_datetime_now() -> str:
	"""Get the current datetime for now.

	Returns:
	    str:
	        The current datetime.
	"""
	return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def get_simulation_uuid() -> str:
	"""Get a new simulation uuid.

	Returns:
	    str:
	        The simulation uuid.
	"""
	simulation_uuid = str(uuid.uuid4())
	return simulation_uuid


def get_examples_outdir() -> str:
	"""Get the output directory for calibration examples.

	Returns:
	    str:
	        The output directory.
	"""
	return osp.join("examples", "outdir")

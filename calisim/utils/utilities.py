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
	    str: The current datetime.
	"""
	return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def get_simulation_uuid() -> str:
	"""Get a new simulation uuid.

	Returns:
	    str: The simulation uuid.
	"""
	simulation_uuid = str(uuid.uuid4())
	return simulation_uuid


def get_examples_outdir() -> str:
	"""Get the output directory for calibration examples.

	Returns:
	    str: The output directory.
	"""
	return osp.join("examples", "outdir")


class EarlyStopper:
	"""Early stopping for training."""

	def __init__(self, patience: int = 1, min_delta: float = 0):
		"""EarlyStopper constructor.

		Args:
		    patience (int, optional): The number of iterations
		           before performing early stopping. Defaults to 1.
		    min_delta (float, optional): The minimum difference
		       for the validation loss. Defaults to 0.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter: int = 0
		self.min_validation_loss = float("inf")

	def early_stop(self, validation_loss: float) -> bool:
		"""Perform early stopping.

		Args:
		    validation_loss (float): The training validation loss.

		Returns:
		    bool: Whether to perform early stopping.
		"""
		if validation_loss < self.min_validation_loss:
			self.min_validation_loss = validation_loss
			self.counter = 0
		elif validation_loss > (self.min_validation_loss + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False

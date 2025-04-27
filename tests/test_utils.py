"""
Tests for various utility functions.

A battery of tests to validate the utility functions.

"""

from pathlib import Path

import numpy as np
from pytest_mock import MockerFixture

from calisim.utils import (
	EarlyStopper,
	create_file_path,
	extend_X,
	get_datetime_now,
	get_examples_outdir,
	get_simulation_uuid,
)


def test_get_datetime_now() -> None:
	time_now = get_datetime_now()
	time_future = get_datetime_now()
	assert time_future == time_now


def test_get_simulation_uuid() -> None:
	simulation_id_1 = get_simulation_uuid()
	simulation_id_2 = get_simulation_uuid()
	assert simulation_id_1 != simulation_id_2


def test_get_examples_outdir() -> None:
	outdir = get_examples_outdir()
	assert outdir == "examples/outdir"


def test_create_file_path(mocker: MockerFixture, outdir: str) -> None:
	file_path = str(Path(outdir, "test_file"))
	mocker.patch(
		"calisim.utils.utilities.create_file_path",
		return_value=file_path,
	)

	created_file_path = create_file_path(file_path)
	assert created_file_path == file_path


def test_extend_X() -> None:
	X = np.array([[1, 2, 3], [4, 5, 6]])
	X_extended = extend_X(X, 2)
	assert X_extended.shape[1] > X.shape[1]


def test_early_stopper_true() -> None:
	early_stopper = EarlyStopper(patience=1, min_delta=0)
	early_stop = False
	for _ in range(5):
		early_stop = early_stopper.early_stop(1)
	assert early_stop


def test_early_stopper_false() -> None:
	early_stopper = EarlyStopper(patience=1, min_delta=0)
	early_stop = early_stopper.early_stop(1)
	assert not early_stop

"""
This module tests the `connect_x.utils` module.
"""

import pytest
import numpy as np

from connect_x import utils


@pytest.mark.parametrize(
    "array,window_size,windows",
    [
        ([1, 2, 3, 4], 2, [[1, 2], [2, 3], [3, 4]]),
        ([1, 2, 3, 4], 3, [[1, 2, 3], [2, 3, 4]]),
    ],
)
def test_rolling_window(array, window_size, windows):
    assert (utils.rolling_window(array, window_size) == windows).all()


@pytest.mark.parametrize(
    "matrix,diagonals",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [
                [10],
                [7, 11],
                [4, 8, 12],
                [1, 5, 9],
                [2, 6],
                [3],
                [1],
                [4, 2],
                [7, 5, 3],
                [10, 8, 6],
                [11, 9],
                [12],
            ],
        )
    ],
)
def test_matrix_diagonals(matrix, diagonals):
    for actual, target in zip(utils.matrix_diagonals(matrix), diagonals):
        assert (actual == target).all()


@pytest.mark.parametrize(
    "matrix,rows",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        )
    ],
)
def test_matrix_rows(matrix, rows):
    for actual, target in zip(utils.matrix_rows(matrix), rows):
        assert (actual == target).all()


@pytest.mark.parametrize(
    "matrix,columns",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
        )
    ],
)
def test_matrix_rows(matrix, columns):
    for actual, target in zip(utils.matrix_columns(matrix), columns):
        assert (actual == target).all()

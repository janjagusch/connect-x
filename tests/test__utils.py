"""
This module tests the `connect_x.utils` module.
"""

import pytest
import numpy as np

from connect_x import utils


# pylint: disable=protected-access


@pytest.mark.parametrize(
    "board,n_rows,n_cols,matrix",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8], 2, 4, [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ([1, 2, 3, 4, 5, 6, 7, 8], 4, 2, [[1, 2], [3, 4], [5, 6], [7, 8]]),
    ],
)
def test_board_to_matrix(board, n_rows, n_cols, matrix):
    np.testing.assert_array_equal(utils.board_to_matrix(board, n_rows, n_cols), matrix)


@pytest.mark.parametrize(
    "matrix,board", [([[1, 2, 3, 4], [5, 6, 7, 8]], [1, 2, 3, 4, 5, 6, 7, 8])]
)
def test_matrix_to_board(matrix, board, to_array):
    assert utils.matrix_to_board(to_array(matrix)) == board


@pytest.mark.parametrize(
    "array,window_size,windows",
    [
        ([1, 2, 3, 4], 2, [[1, 2], [2, 3], [3, 4]]),
        ([1, 2, 3, 4], 3, [[1, 2, 3], [2, 3, 4]]),
    ],
)
def test_rolling_window(array, window_size, windows, to_array):
    np.testing.assert_array_equal(
        utils.rolling_window(to_array(array), window_size), windows
    )


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
def test_matrix_diagonals(matrix, diagonals, to_array):
    for actual, target in zip(utils.matrix_diagonals(to_array(matrix)), diagonals):
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
def test_matrix_rows(matrix, rows, to_array):
    np.testing.assert_array_equal(utils.matrix_rows(to_array(matrix)), rows)


@pytest.mark.parametrize(
    "matrix,columns",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
        )
    ],
)
def test_matrix_cols(matrix, columns, to_array):
    np.testing.assert_array_equal(utils.matrix_columns(to_array(matrix)), columns)


@pytest.mark.parametrize("mark,other_mark", [(1, 2), (2, 1), (None, None),])
def test_other_mark(mark, other_mark):
    assert utils.other_mark(mark) == other_mark

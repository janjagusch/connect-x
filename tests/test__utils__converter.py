"""
This module tests the `connect_x.utils.converter` module.
"""

import numpy as np
import pytest

from connect_x.utils import converter


@pytest.fixture(name="matrix_example")
def matrix_example_():
    """
    Example from: https://github.com/denkspuren/BitboardC4/blob/master/
    BitboardDesign.md#using-two-longs-to-encode-the-board.
    """
    matrix = np.zeros((6, 7)).astype(int)
    matrix[-1, 2] = 1
    matrix[-1, 3] = 2
    matrix[-2, 3] = 2
    matrix[-3, 3] = 1
    matrix[-1, 4] = 1
    matrix[-2, 4] = 2
    return matrix


@pytest.fixture(name="board_example")
def board_example_(matrix_example):
    """
    Example from: https://github.com/denkspuren/BitboardC4/blob/master/
    BitboardDesign.md#using-two-longs-to-encode-the-board.
    """
    return converter.matrix_to_board(matrix_example)


@pytest.fixture(name="bitmaps_example")
def bitmaps_example_():
    """
    Example from: https://github.com/denkspuren/BitboardC4/blob/master/
    BitboardDesign.md#using-two-longs-to-encode-the-board.
    """
    return [276840448, 543162368]


@pytest.mark.parametrize(
    "board,rows,columns,matrix",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ([1, 2, 3, 4, 5, 6], 2, 3, [[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_board_to_matrix(board, rows, columns, matrix):
    np.testing.assert_array_equal(
        converter.board_to_matrix(board, rows, columns), matrix
    )


@pytest.mark.parametrize(
    "matrix,board", [([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),]
)
def test_matrix_to_board(matrix, board, to_array):
    assert converter.matrix_to_board(to_array(matrix)) == board


def test_board_to_bitmaps_empty(board, configuration):
    assert converter.board_to_bitmaps(
        board, configuration.rows, configuration.columns
    ) == [0, 0]


def test_board_to_bitmaps_example(board_example, bitmaps_example, configuration):
    assert (
        converter.board_to_bitmaps(
            board_example, configuration.rows, configuration.columns
        )
        == bitmaps_example
    )


def test_bitmaps_to_board_empty(board):
    assert converter.bitmaps_to_board([0, 0]) == board


def test_bitmaps_to_board_example(bitmaps_example, board_example):
    assert converter.bitmaps_to_board(bitmaps_example) == board_example

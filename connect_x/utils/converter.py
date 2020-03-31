"""
This module provides functions to represent boards as bitmaps, to check whether for
tokens are connected and to make a move.
The code is taken from [Gilles Vandewiele's Medium Post](https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0). Adjustment have been made
based upon [Dominikus Herzberg's GitHub post](https://github.com/denkspuren/BitboardC4/blob/master/BitboardDesign.md).
"""

import numpy as np


def board_to_matrix(board, rows, columns):
    return np.array(board).reshape(rows, columns)


def _matrix_to_array(matrix):
    matrix = np.insert(matrix, 0, 0, axis=0)
    return np.rot90(matrix).reshape(1, -1)[0]


def _split_array(array):
    def _bin_array(array, token):
        assert token in (1, 2)
        bin_array = np.zeros(array.shape)
        bin_array[array == token] = 1
        return bin_array.astype(int)

    return _bin_array(array, 1), _bin_array(array, 2)


def _array_to_bitmap(array):
    return int("".join(map(str, array)), 2)


def board_to_bitmaps(board):
    matrix = board_to_matrix(board, 6, 7)
    array = _matrix_to_array(matrix)
    array_me, array_other = _split_array(array)
    return [_array_to_bitmap(array_me), _array_to_bitmap(array_other)]


def _bitmap_to_array(bitmap):
    array = [int(bit) for bit in bin(bitmap)[2:]]
    array = [0 for _ in range(49 - len(array))] + array
    return np.array(array)


def _merge_arrays(array_me, array_other):
    assert array_me.shape == array_other.shape
    array = np.zeros(array_me.shape)

    array[array_me == 1] = 1
    array[array_other == 1] = 2

    return array.astype(int)


def _array_to_matrix(array):
    matrix = array.reshape(7, 7)[:, 1:]
    return np.rot90(matrix, k=3)


def matrix_to_board(matrix):
    return matrix.reshape(1, -1).tolist()[0]


def bitmaps_to_board(bitmaps):
    matrix = bitmaps_to_matrix(bitmaps)
    board = matrix_to_board(matrix)
    return board


def bitmaps_to_matrix(bitmaps):
    bitmap_me, bitmap_other = bitmaps
    array_me = _bitmap_to_array(bitmap_me)
    array_other = _bitmap_to_array(bitmap_other)
    array = _merge_arrays(array_me, array_other)
    matrix = _array_to_matrix(array)
    return matrix

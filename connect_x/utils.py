"""
This module contains useful functions for this project.
"""

import numpy as np


def game_round(matrix):
    """
    Returns in which round the game is.

    Args:
        matrix (np.array): The board as a matrix.

    Returns:
        int: The round of the game.
    """
    return (matrix != 0).sum()


def middle_column(matrix):
    """
    Returns the index of the middle column of the boards.

    Args:
        matrix (np.array): The board as a matrix.

    Returns:
        int: The index of the middle column.
    """
    _, columns = matrix.shape
    return int(np.floor(columns / 2))


def other_mark(mark):
    """
    Given the mark of a token, returns the other mark.

    Args:
        mark (int): The mark of the token.

    Returns:
        int: The other mark or `None`, when mark is `None`.
    """
    if not mark:
        return None
    assert mark in (1, 2)
    return 2 if mark == 1 else 1


def board_to_matrix(board, n_rows, n_cols):
    """
    Converts a board into a numpy matrix.

    Args:
        board (list): The board state.
        n_rows (int): Number of rows on the board.
        n_cols (int): Number of columns on the board.

    Returns:
        np.array: The board as a matrix.
    """
    return np.array(board).reshape(n_rows, n_cols)


def matrix_to_board(matrix):
    """
    Converts a matrix into a board.

    Args:
        matrix (np.array): The board matrix.

    Returns:
        list: The board as a list.
    """
    return matrix.reshape(1, -1).tolist()[0]


def rolling_window(array, window_size):
    """
    Returns rolling windows over a 1-dimensional array.

    Args:
        array (np.array): A 1-dimensional arary.
        window_size (int): The window size.

    Returns:
        list: List of np.array objects.
    """
    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def _diagonals(matrix):
    return [matrix.diagonal(i) for i in range(-matrix.shape[0] + 1, matrix.shape[1])]


def matrix_diagonals(matrix):
    """
    Returns all diagonals of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return _diagonals(matrix) + _diagonals(matrix[::-1])


def matrix_rows(matrix):
    """
    Returns all rows of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return list(matrix)


def matrix_columns(matrix):
    """
    Returns all columns of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return list(matrix.T)

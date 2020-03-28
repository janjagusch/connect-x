"""
This module contains useful functions for this project.
"""

import numpy as np


TOKEN_ME = 1
TOKEN_OTHER = 9


def matrix_hash(matrix):
    """
    Returns a hash representation of the matrix.
    Useful for using it as keys in dictionaries.

    Args:
        matrix (np.array): The board state as matrix.

    Returns:
        str: The matrix hash.
    """
    return matrix.tostring().decode("utf8")


def mark_agnostic_board(board, mark):
    """
    Makes the board mark agnostic. Replaces your mark with `1` and the other mark with
    `9`.

    Args:
        board (list): The board state.
        mark (int): Your token mark.

    Returns:
        list: The mark agnostic board.
    """

    def agnostic(val, mark):
        if val == 0:
            return val
        return TOKEN_ME if val == mark else TOKEN_OTHER

    return [agnostic(val, mark) for val in board]


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


def other_token(token):
    """
    Given a token, returns the other token.

    Args:
        token (int): The token.

    Returns:
        int: The other token or `None`.
    """
    if not token:
        return None
    assert token in (TOKEN_ME, TOKEN_OTHER)
    return TOKEN_OTHER if token == TOKEN_ME else TOKEN_ME


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

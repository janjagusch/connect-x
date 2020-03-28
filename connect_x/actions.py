"""
This module provides functions to determine possible actions given an observation
and execute those actions.
"""

import numpy as np

from . import utils


def possible_actions(matrix):
    """
    Returns all possible actions you can take from a matrix state.

    Args:
        matrix (np.array): The board matrix.

    Returns:
        list: The possible actions.
    """
    filter_ = [(array == 0).any() for array in utils.board.matrix_columns(matrix)]
    return list(np.arange(matrix.shape[1])[filter_])


def step(matrix, action, token):
    """
    Throws a token into a column of the board.

    Args:
        matrix (np.array): The board matrix.
        action (int): The column index where the token should be placed.
        token (int): The token.

    Returns:
        np.array: The new token matrix.
    """
    col = matrix[:, action]
    row = np.argwhere(col == 0).max()
    new_matrix = matrix.copy()
    new_matrix[row, action] = token
    return new_matrix

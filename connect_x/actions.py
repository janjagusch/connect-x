"""
This module provides functions to determine possible actions given an observation
and execute those actions.
"""

from copy import copy
import numpy as np

from . import utils


def _possible_actions(matrix):
    """
    Returns all possible actions you can take from a matrix state.

    Args:
        matrix (np.array): The board matrix.

    Returns:
        np.array: The possible actions.
    """
    filter_ = [(array == 0).any() for array in utils.matrix_columns(matrix)]
    return np.arange(matrix.shape[1])[filter_]


def _step(matrix, action, mark):
    """
    Applies an action with a mark to a matrix.

    Args:
        matrix (np.array): The board matrix.
        action (int): The column index where the token should be placed.
        mark (int): The mark of the token.

    Returns:
        np.array: The new token matrix.
    """
    col = matrix[:, action]
    row = np.argwhere(col == 0).max()
    new_matrix = matrix.copy()
    new_matrix[row, action] = mark
    return new_matrix


def possible_actions(observation, configuration):
    """
    Lists all possible actions that can be taken.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.

    Returns:
        list: List of possible actions.
    """
    board = observation.board
    n_rows = configuration.rows
    n_cols = configuration.columns

    matrix = utils.board_to_matrix(board, n_rows, n_cols)
    return list(_possible_actions(matrix))


def step(observation, configuration, action, mark):
    """
    Executes an action and returns the new observation.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.
        action (int): The index of the column where you want to insert the token.
        mark (int): The mark of the token.

    Returns:
        dict: The new observation.
    """
    board = observation.board
    n_rows = configuration.rows
    n_cols = configuration.columns

    matrix = utils.board_to_matrix(board, n_rows, n_cols)
    new_matrix = _step(matrix, action, mark)

    new_board = utils.matrix_to_board(new_matrix)

    new_observation = copy(observation)
    new_observation.board = new_board

    return new_observation

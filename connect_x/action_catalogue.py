"""
This module contains hard-coded rules for Connect X.
"""

import numpy as np

from .utils.converter import bitmaps_to_matrix


def _middle_column(matrix):
    return np.floor(matrix.shape[1] / 2).astype(int)


# pylint: disable=unused-argument
def _action_0(matrix, mark):
    return _middle_column(matrix)


def _action_1(matrix, mark):
    middle_column = _middle_column(matrix)
    if matrix[-1, middle_column] != 0:
        return middle_column + 1
    return middle_column


def _action_2(matrix, mark):
    middle_column = _middle_column(matrix)
    if (
        matrix[-1, middle_column] == mark
        and matrix[-1, middle_column + 1] == 0
        and matrix[-1, middle_column - 1] == 0
        and matrix[-1, middle_column - 2] == 0
        and matrix[-1, middle_column + 2] == 0
    ):
        return middle_column + 1
    return None


# pylint: enable=unused-argument


_ACTION_CATALOGUE = {
    0: _action_0,
    1: _action_1,
    2: _action_2,
}


def get_action(state, player):
    """
    Returns an action from the _ACTION_CATALOGUE, given the state.

    Args:
        state (connect_x.game.connect_x.ConnectXState): The state.
        player (int): The player.

    Returns:
        int: The action.
    """
    matrix = bitmaps_to_matrix(state.bitmaps)
    mark = player + 1
    action_func = _ACTION_CATALOGUE.get(state.counter)
    if action_func:
        return action_func(matrix, mark)
    return None

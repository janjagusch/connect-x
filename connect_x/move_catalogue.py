"""
This module contains hard-coded rules for Connect X.
"""

from .utils.board import (
    middle_column,
    game_round,
    board_to_matrix,
    TOKEN_ME,
)


# pylint: disable=unused-argument
def _move_0(matrix, **kwargs):
    """
    If you start the game and are first player, always choose the middle column.
    """
    return middle_column(matrix)


def _move_1(matrix, **kwargs):
    """
    If you start the game and are the second player, make sure not to fall for the
    cheap trick.
    """
    middle_column_ = middle_column(matrix)
    if matrix[-1, middle_column_] != 0:
        return middle_column_ + 1
    return middle_column_


def _move_2(matrix, **kwargs):
    """
    If you are the first player and it is your second turn, see if you can go for the
    cheap trick.
    """
    middle_column_ = middle_column(matrix)

    if (
        matrix[-1, middle_column_] == TOKEN_ME
        and matrix[-1, middle_column_ + 1] == 0
        and matrix[-1, middle_column_ - 1] == 0
        and matrix[-1, middle_column_ - 2] == 0
        and matrix[-1, middle_column_ + 2] == 0
    ):
        return middle_column_ + 1
    return None


# pylint: enable=unused-argument


_MOVE_CATALOGUE = {
    0: _move_0,
    1: _move_1,
    2: _move_2,
    3: _move_2,
}


def move(matrix, configuration):
    """
    Makes a move based on the `_MOVE_CATALOGUE`.

    Args:
        matrix (np.array): The board state as matrix.
        configuration (kaggle_environments.utils.Struct): The configuration.

    Return:
        int: The action.
    """

    move_func = _MOVE_CATALOGUE.get(game_round(matrix))
    if move_func:
        return move_func(matrix, configuration=configuration)
    return None

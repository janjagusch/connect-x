"""
This module contains hard-coded rules for Connect X.
"""

from . import utils


# pylint: disable=unused-argument
def _move_0(matrix, **kwargs):
    """
    If you start the game and are first player, always choose the middle column.
    """
    return utils.board.middle_column(matrix)


def _move_1(matrix, **kwargs):
    """
    If you start the game and are the second player, make sure not to fall for the
    cheap trick.
    """
    middle_column = utils.board.middle_column(matrix)
    if matrix[-1, middle_column] != 0:
        return middle_column + 1
    return middle_column


# pylint: enable=unused-argument


def _move_2(matrix, **kwargs):
    """
    If you are the first player and it is your second turn, see if you can go for the
    cheap trick.
    """
    middle_column = utils.board.middle_column(matrix)
    mark = kwargs["mark"]

    if (
        matrix[-1, middle_column] == mark
        and matrix[-1, middle_column + 1] == 0
        and matrix[-1, middle_column - 1] == 0
        and matrix[-1, middle_column - 2] == 0
        and matrix[-1, middle_column + 2] == 0
    ):
        return middle_column + 1
    return None


def _move_n(matrix, **kwargs):
    """
    Try to complete the cheap trick.
    """
    middle_column = utils.board.middle_column(matrix)
    mark = kwargs["mark"]

    if (
        matrix[-1, middle_column] == mark
        and matrix[-1, middle_column + 1] == mark
        and matrix[-1, middle_column - 1] == 0
        and matrix[-1, middle_column - 2] == 0
        and matrix[-1, middle_column + 2] == 0
    ):
        return middle_column - 1
    return None


_MOVE_CATALOGUE = {
    0: _move_0,
    1: _move_1,
    2: _move_2,
    3: _move_2,
}


def move(observation, configuration):
    """
    Makes a move based on the `_MOVE_CATALOGUE`.

    Args:
        observation (kaggle_environments.utils.board.Struct): The observation.
        configuration (kaggle_environments.utils.board.Struct): The configuration.

    Return:
        int: The action.
    """
    board = observation.board
    mark = observation.mark
    rows = configuration.rows
    columns = configuration.columns
    matrix = utils.board.board_to_matrix(board, rows, columns)
    game_round = utils.board.game_round(matrix)

    move_func = _MOVE_CATALOGUE.get(game_round)
    if move_func:
        return move_func(matrix, mark=mark, configuration=configuration)
    return None

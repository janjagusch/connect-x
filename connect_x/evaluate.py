"""
This module contains functions to evaluate environment observations.
"""

import numpy as np

from . import utils


def _windows(matrix, window_size):
    """
    Calculates all windows that are relevant to evaluate to board state from a matrix.

    Args:
        matrix (np.array): A board matrix.
        window_size (int): The number of token you need to have 'in a row'.

    Returns:
        np.array: The windows of the board.
    """
    windows = []
    # pylint: disable=bad-continuation
    for array in (
        utils.matrix_rows(matrix)
        + utils.matrix_columns(matrix)
        + utils.matrix_diagonals(matrix)
    ):
        # pylint: enable=bad-continuation
        if len(array) >= window_size:
            windows.extend(utils.rolling_window(array, window_size))
    return np.array(windows)


def _eval_windows(windows, mark):
    """
    Calculates the evaluation windows, depending on the mark of the token.

    Args:
        windows (np.array): Array of windows.
        mark (int): `1` or `2`.

    Returns:
        np.array: Array of evaluation windows.
    """
    mark_opponent = 2 if mark == 1 else 1
    eval_windows = np.zeros(windows.shape)
    eval_windows[windows == mark] = 1
    eval_windows[windows == mark_opponent] = -1
    return eval_windows


def _evaluate_victory(eval_windows):
    """
    Checks whether evaluation windows contain a victory.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: Whether evaluation windows contain victory.
    """
    return (eval_windows.mean(axis=1) == 1).any()


def _evaluate_board_full(eval_windows):
    """
    Checks whether the board is full.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: Whether the board is full.
    """

    return not (eval_windows == 0).any()


def _evaluate_heuristic(eval_windows):
    """
    Evaluates the board.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: The value of the board.
    """
    values = np.exp2(eval_windows.sum(axis=1))
    not_contains_other = eval_windows.min(axis=1) != -1
    return (values * not_contains_other).mean()


def _evaluate(eval_windows):
    """
    Evaluates the board. Calculates a value for the board and checks whether the game
    has ended.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        tuple: (The value of the board, Whether the game has ended).
    """
    if _evaluate_victory(eval_windows):
        return float("inf"), True
    if _evaluate_board_full(eval_windows):
        return float(0), True
    return _evaluate_heuristic(eval_windows), False


def evaluate(observation, configuration):
    """
    Evaluates an observation.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.

    Returns:
        tuple: (The value of the board, Whether the game has ended).
    """
    mark = observation.mark
    mark_opponent = 2 if mark == 1 else 1

    matrix = utils.board_to_matrix(
        observation.board, configuration.rows, configuration.columns
    )
    windows = _windows(matrix, configuration.inarow)
    eval_windows = _eval_windows(windows, mark)
    eval_windows_opponent = _eval_windows(windows, mark_opponent)
    value, done = _evaluate(eval_windows)
    value_opponent, done_opponent = _evaluate(eval_windows_opponent)
    return value - value_opponent, any([done, done_opponent])

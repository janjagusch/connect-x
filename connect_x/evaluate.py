"""
This module contains functions to evaluate environment observations.
"""

import numpy as np

from .utils.board import (
    TOKEN_ME,
    TOKEN_OTHER,
    matrix_rows,
    matrix_columns,
    matrix_diagonals,
    rolling_window,
)


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

    for array in (
        matrix_rows(matrix) + matrix_columns(matrix) + matrix_diagonals(matrix)
    ):
        if len(array) >= window_size:
            windows.extend(rolling_window(array, window_size))
    return np.array(windows)


def _eval_windows(windows, mark_me, mark_other):
    """
    Calculates the evaluation windows.

    Args:
        windows (np.array): Array of windows.
        mark_me (int): The mark of my player.
        mark_other (int): The mark of the other player.

    Returns:
        np.array: Array of evaluation windows.
    """
    eval_windows = np.zeros(windows.shape)
    eval_windows[windows == mark_me] = 1
    eval_windows[windows == mark_other] = -1
    return eval_windows.astype(int)


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


def evaluate(matrix, configuration):
    """
    Evaluates an observation.

    Args:
        matrix (dict): The matrix of the board state.
        configuration (dict): The configuration.

    Returns:
        tuple: (The value of the board, Whether the game has ended).
    """
    windows = _windows(matrix, configuration.inarow)
    eval_windows_me = _eval_windows(windows, TOKEN_ME, TOKEN_OTHER)
    eval_windows_other = _eval_windows(windows, TOKEN_OTHER, TOKEN_ME)
    value_me, done_me = _evaluate(eval_windows_me)
    value_other, done_other = _evaluate(eval_windows_other)
    return value_me - value_other, any([done_me, done_other])

"""
This module contains useful functions for this project.
"""

import numpy as np


def rolling_window(array, window_size):
    """
    Returns rolling windows over a 1-dimensional array.

    Args:
        array (np.array): A 1-dimensional arary.
        window_size (int): The window size.

    Retuns:
        list: List of np.array objects.
    """
    assert window_size > 0, "window_size must be > 0."
    assert window_size <= len(array), "window_size must be <= len(array)"
    if isinstance(array, list):
        array = np.array(array)
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
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return _diagonals(matrix) + _diagonals(matrix[::-1])


def matrix_rows(matrix):
    """
    Returns all rows of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return [matrix[row_index, :] for row_index in range(matrix.shape[0])]


def matrix_columns(matrix):
    """
    Returns all columns of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return [matrix[:, column_index] for column_index in range(matrix.shape[1])]

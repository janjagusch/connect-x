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

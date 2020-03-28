"""
This module tests the `connect_x.actions` module.
"""

import pytest
import numpy as np

from connect_x import actions


@pytest.mark.parametrize(
    "matrix,possible_actions",
    [([[0, 0, 0], [0, 0, 0]], [0, 1, 2]), ([[1, 0, 0], [1, 0, 0]], [1, 2]),],
)
def test_possible_actions(matrix, possible_actions, to_array):

    assert actions.possible_actions(to_array(matrix)) == possible_actions


@pytest.mark.parametrize(
    "matrix,action,token,new_matrix",
    [
        ([[0, 0, 0], [0, 0, 0]], 0, 1, [[0, 0, 0], [1, 0, 0]]),
        ([[0, 0, 0], [0, 0, 1]], 2, 1, [[0, 0, 1], [0, 0, 1]]),
        ([[0, 0, 0], [0, 0, 1]], 2, 2, [[0, 0, 2], [0, 0, 1]]),
    ],
)
def test_step(
    matrix, action, token, new_matrix, to_array,
):

    np.testing.assert_array_equal(
        actions.step(to_array(matrix), action, token), new_matrix
    )

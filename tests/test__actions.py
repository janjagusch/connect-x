"""
This module tests the `connect_x.actions` module.
"""

import pytest

from connect_x import actions, utils


@pytest.mark.parametrize(
    "matrix,rows,columns,possible_actions",
    [
        ([[0, 0, 0], [0, 0, 0]], 2, 3, [0, 1, 2]),
        ([[1, 0, 0], [1, 0, 0]], 2, 3, [1, 2]),
    ],
)
def test_possible_actions(
    matrix, rows, columns, observation, configuration, possible_actions, to_array
):
    board = utils.matrix_to_board(to_array(matrix))

    observation.board = board
    configuration.rows = rows
    configuration.columns = columns

    assert actions.possible_actions(observation, configuration) == possible_actions


@pytest.mark.parametrize(
    "matrix,rows,columns,action,mark,new_matrix",
    [
        ([[0, 0, 0], [0, 0, 0]], 2, 3, 0, 1, [[0, 0, 0], [1, 0, 0]]),
        ([[0, 0, 0], [0, 0, 1]], 2, 3, 2, 1, [[0, 0, 1], [0, 0, 1]]),
        ([[0, 0, 0], [0, 0, 1]], 2, 3, 2, 2, [[0, 0, 2], [0, 0, 1]]),
    ],
)
def test_step(
    matrix,
    rows,
    columns,
    observation,
    configuration,
    action,
    mark,
    new_matrix,
    to_array,
):
    board = utils.matrix_to_board(to_array(matrix))
    new_board = utils.matrix_to_board(to_array(new_matrix))

    observation.board = board
    configuration.rows = rows
    configuration.columns = columns

    new_observation = actions.step(observation, configuration, action, mark)
    assert isinstance(new_observation, observation.__class__)
    assert new_observation.board == new_board

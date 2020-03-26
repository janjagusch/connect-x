"""
This module tests the `connect_x.evaluate` module.
"""

import pytest
import numpy as np

from connect_x import evaluate


# pylint: disable=protected-access


@pytest.mark.parametrize(
    "matrix,window_size,windows",
    [
        (
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                2,
                [
                    [1, 2],
                    [2, 3],
                    [4, 5],
                    [5, 6],
                    [7, 8],
                    [8, 9],
                    [1, 4],
                    [4, 7],
                    [2, 5],
                    [5, 8],
                    [3, 6],
                    [6, 9],
                    [4, 8],
                    [1, 5],
                    [5, 9],
                    [2, 6],
                    [4, 2],
                    [7, 5],
                    [5, 3],
                    [8, 6],
                ],
            ]
        )
    ],
)
def test__windows(matrix, window_size, windows, to_array):
    np.testing.assert_array_equal(
        evaluate._windows(to_array(matrix), window_size), windows
    )


@pytest.mark.parametrize(
    "windows,mark,eval_windows",
    [
        (
            [[1, 1], [1, 0], [0, 0], [1, 2], [2, 2]],
            1,
            [[1, 1], [1, 0], [0, 0], [1, -1], [-1, -1]],
        ),
        (
            [[1, 1], [1, 0], [0, 0], [1, 2], [2, 2]],
            2,
            [[-1, -1], [-1, 0], [0, 0], [-1, 1], [1, 1]],
        ),
    ],
)
def test__eval_windows(windows, mark, eval_windows, to_array):
    np.testing.assert_array_equal(
        evaluate._eval_windows(to_array(windows), mark), eval_windows
    )


@pytest.mark.parametrize(
    "eval_windows,victory", [([[1, 1, 1]], True), ([[1, 1, 0]], False)]
)
def test__evaluate_victory(eval_windows, victory, to_array):
    np.testing.assert_array_equal(
        evaluate._evaluate_victory(to_array(eval_windows)), victory
    )


@pytest.mark.parametrize("eval_windows,full", [([1, 2, 1], True), ([1, 1, 0], False)])
def test__evaluate_board_full(eval_windows, full, to_array):
    np.testing.assert_array_equal(
        evaluate._evaluate_board_full(to_array(eval_windows)), full
    )


@pytest.mark.parametrize(
    "eval_windows,result",
    [([[1, 1, 1]], (float("inf"), True)), ([[1, -1, 1]], (0, True))],
)
def test__evaluate(eval_windows, result, to_array):
    assert evaluate._evaluate(to_array(eval_windows)) == result


@pytest.mark.parametrize("eval_windows", [([[1, 1, 0]]),])
def test__evaluate_with_heuristics(eval_windows, to_array):
    result = evaluate._evaluate(to_array(eval_windows))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert result[0] >= 0
    assert isinstance(result[1], bool)


def test_evaluate(env):
    observation = env.state[0].observation
    configuration = env.configuration
    result = evaluate.evaluate(observation, configuration)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], bool)

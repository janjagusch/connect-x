"""
This module provides tests for the `connect_x.simulator` module.
"""

import numpy as np
import pytest

from connect_x.simulator import Simulator, is_win
from connect_x.utils import board_converter


@pytest.fixture(name="simulator", scope="function")
def simulator_(bitmaps):
    return Simulator(bitmaps)


@pytest.fixture(name="matrix_win")
def matrix_win_():
    matrix = np.zeros((6, 7)).astype(int)
    matrix[-1, 0] = 1
    matrix[-2, 0] = 1
    matrix[-3, 0] = 1
    matrix[-4, 0] = 1
    return matrix


@pytest.fixture(name="board_win")
def board_win_(matrix_win):
    return board_converter.matrix_to_board(matrix_win)


@pytest.fixture(name="bitmap_win")
def bitmap_win_(board_win):
    return board_converter.board_to_bitmaps(board_win, 1, 2)[0]


def test_action_log(simulator):
    assert simulator.action_log == []


def test_height(simulator):
    assert simulator.height == [0, 7, 14, 21, 28, 35, 42]


def test_counter(simulator):
    assert simulator.counter == 0


def test_valid_actions(simulator):
    assert simulator.valid_actions == [0, 1, 2, 3, 4, 5, 6]


def test_do_action(simulator):
    simulator.do_action(0)
    assert simulator.bitmaps[0] == 1
    assert simulator.height[0] == 1
    assert simulator.action_log == [0]
    assert simulator.counter == 1


def test_undo_action(simulator):
    simulator.do_action(0)
    simulator.undo_action()
    assert simulator.bitmaps[0] == 0
    assert simulator.height[0] == 0
    assert simulator.action_log == []
    assert simulator.counter == 0


def test_is_win_false(simulator):
    assert not is_win(simulator.bitmaps[0])
    assert not is_win(simulator.bitmaps[1])


def test_is_win_true(bitmap_win):
    assert is_win(bitmap_win)

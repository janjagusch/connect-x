"""
This module provides tests for the `connect_x.board_action_map` module.
"""

from connect_x.matrix_action_map import MATRIX_ACTION_MAP


def test_board_action_map():
    assert isinstance(MATRIX_ACTION_MAP, dict)
    assert len(MATRIX_ACTION_MAP) > 0
    assert all([isinstance(key, str) for key in MATRIX_ACTION_MAP.keys()])
    assert all([isinstance(value, int) for value in MATRIX_ACTION_MAP.values()])

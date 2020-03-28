"""
This module provides tests for the `connect_x.board_action_map` module.
"""

from connect_x import board_action_map


def test_board_action_map():
    assert isinstance(board_action_map.BOARD_ACTION_MAP, dict)
    assert len(board_action_map.BOARD_ACTION_MAP) > 0
    assert all(
        [isinstance(key, str) for key in board_action_map.BOARD_ACTION_MAP.keys()]
    )
    assert all(
        [isinstance(value, int) for value in board_action_map.BOARD_ACTION_MAP.values()]
    )

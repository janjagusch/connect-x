"""
This module contains the submissoin for the Kaggle competition.
"""

from connect_x.minimax import minimax, ConnectXNode
from connect_x.move_catalogue import move
from connect_x.board_action_map import (
    FORECAST_DEPTH as PRECOMPUTED_DEPTH,
    BOARD_ACTION_MAP,
)
from connect_x.utils.board import (
    game_round,
    mark_agnostic_board,
    matrix_hash,
    board_to_matrix,
)
from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)

FORECAST_DEPTH = 3


def _rule_based_action(matrix, configuration):
    return move(matrix, configuration)


# pylint: disable=unused-argument
def _precomputed_action(matrix, configuration):
    if PRECOMPUTED_DEPTH > (game_round(matrix) + FORECAST_DEPTH):
        return BOARD_ACTION_MAP[matrix_hash(matrix)]
    return None


# pylint: enable=unused-argument


def _forecasted_action(matrix, configuration):
    node = ConnectXNode(matrix, configuration)
    next_node, _ = minimax(node, max_depth=FORECAST_DEPTH)
    return next_node.action


def act(observation, configuration):
    """
    Decides what action to do next.

    Args:
        observation (kaggle_environments.utils.Struct): The observation.
        configuration (kaggle_environments.utils.Struct): The configuration.

    Returns:
        int: The action.
    """
    board = mark_agnostic_board(observation.board, observation.mark)
    matrix = board_to_matrix(board, configuration.rows, configuration.columns)
    action = (
        _rule_based_action(matrix, configuration)
        or _precomputed_action(matrix, configuration)
        or _forecasted_action(matrix, configuration)
    )
    return int(action)

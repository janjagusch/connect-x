"""
This module contains the submissoin for the Kaggle competition.
"""

import numpy as np

from connect_x.minimax import minimax, ConnectXNode
from connect_x.move_catalogue import move
from connect_x import board_action_map
from connect_x import utils
from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)

FORECAST_DEPTH = 3


def _game_round(board):
    return (np.array(board) != 0).sum()


def _rule_based_action(observation, configuration):
    return move(observation, configuration)


# pylint: disable=unused-argument
def _precomputed_action(observation, configuration):
    if board_action_map.FORECAST_DEPTH > (
        _game_round(observation.board) + FORECAST_DEPTH
    ):
        board_hash = utils.board.board_hash(observation.board, observation.mark)
        return board_action_map.BOARD_ACTION_MAP[board_hash]
    return None


# pylint: enable=unused-argument


def _forecasted_action(observation, configuration):
    node = ConnectXNode(observation, configuration)
    next_node, _ = minimax(node, max_depth=FORECAST_DEPTH)
    return next_node.action


def act(observation, configuration):
    """
    Decides what action to do next.

    Args:
        observation (kaggle_environments.utils.board.Struct): The observation.
        configuration (kaggle_environments.utils.board.Struct): The configuration.

    Returns:
        int: The action.
    """
    action = (
        _rule_based_action(observation, configuration)
        or _precomputed_action(observation, configuration)
        or _forecasted_action(observation, configuration)
    )
    return int(action)

"""
This module describes the config for the main submission file.
"""

import numpy as np


def heuristic(state, game, player):
    """
    The heuristic function.

    Args:
        state (connect_x.game.connect_x.ConnectXState): The state.
        game (connect_x.game.connect_x.ConnectXGame): The game.

    Returns:
        float: The heuristic value.
    """

    def _heuristic_player(player):
        return np.sum(
            [4 ** (x - 1) * game.connected(state, player, x) for x in range(2, 4)]
        )

    return _heuristic_player(player) - _heuristic_player(1 - player)


def order_actions(actions):
    """
    Orders the actions for the Negamax tree.

    Args:
        actions (list): The list of actions.

    Returns:
        list: The list of prioritized actions.
    """
    order = np.array(actions)
    order = np.absolute(order - 3)
    order = np.argsort(order)
    return np.array(actions)[order]


DEPTH = 3

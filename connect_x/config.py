"""
This module describes the config for the main submission file.
"""

import numpy as np


def connected(state, player, x):
    """
    Returns how many times x are connected in the state of player.
    """
    directions = [1, 6, 7, 8]

    def _connected(bitmap, direction, x):
        assert x > 0
        # pylint: disable=no-member
        return bin(np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)]))[
            2:
        ].count("1")
        # pylint: enable=no-member

    return np.array(
        [_connected(state.bitmaps[player], direction, x) for direction in directions]
    ).sum()


def heuristic(state, player):
    """
    The heuristic function.

    Args:
        state (connect_x.game.connect_x.ConnectXState): The state.
        player (int): The player (0 or 1).

    Returns:
        float: The heuristic value.
    """

    def _heuristic_player(player):
        return np.sum([4 ** (x - 1) * connected(state, player, x) for x in range(2, 4)])

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
    order = np.array(np.argsort(order))
    return np.array(actions)[order]


TIMEOUT_BUFFER = 0.85

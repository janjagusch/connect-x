"""
"""

from copy import deepcopy

import numpy as np


class GameState:
    """
    """

    @property
    def state_hash(self):
        return None


class Game:
    """
    """

    _STATE_CLS = GameState

    def valid_actions(self, state):
        """
        Return a list of the allowable moves at this point.
        """
        raise NotImplementedError

    def do(self, state, action, inplace=False):
        """Return the state that results from making an action from a state."""
        raise NotImplementedError

    def undo(self, state, inplace=False):
        """
        """
        raise NotImplementedError

    @classmethod
    def initial(cls):
        """
        """
        raise NotImplementedError

    def is_win(self, state, player):
        """
        """
        raise NotImplementedError

    def is_end(self, state):
        """
        Warning! This does not check for `is_win`.
        """
        raise NotImplementedError


def x_connected(bitmap, x):
    """
    Counts how many times x tokens are connected in all directions.
    """
    directions = [1, 6, 7, 8]

    def _x_connected(bitmap, x, direction):
        """
        Counts how many times x tokens are connected in one direction.
        """
        assert x > 0
        return bin(
            np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)])
        ).count("1")

    return np.array(
        [_x_connected(bitmap, x, direction) for direction in directions]
    ).sum()


class ConnectXState(GameState):
    """
    """

    def __init__(self, bitmaps, action_log, height, counter):
        self.bitmaps = bitmaps
        self._action_log = action_log
        self._height = height
        self._counter = counter

    @property
    def state_hash(self):
        return 2 * self.bitmaps[0] + self.bitmaps[1]

    def __repr__(self):
        attr_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attr_str})"


class ConnectXGame(Game):
    """
    """

    _STATE_CLS = ConnectXState

    def __init__(self, rows=6, columns=7, x=4):
        assert rows == 6, "The game only supports rows=6 for now."
        assert columns == 7, "The game only supports columns=7 for now."
        assert x == 4, "The game only supports x=4 for now."
        self.rows = rows
        self.columns = columns
        self.x = 4
        self._ACTIONS = range(columns)
        self._TOP = int("_".join("1000000" for _ in range(columns)), 2)

    def valid_actions(self, state):
        return [
            action
            for action in self._ACTIONS
            if not self._TOP & (1 << state._height[action])
        ]

    def do(self, state, action, inplace=False):
        action_bit = 1 << state._height[action]
        if not inplace:
            state = deepcopy(state)
        state.bitmaps[state._counter % 2] ^= action_bit
        state._height[action] += 1
        state._action_log.append(action)
        state._counter += 1
        return state

    def undo(self, state, inplace=False):
        action = state._action_log[-1]
        action_bit = 1 << (state._height[action] - 1)
        if not inplace:
            state = deepcopy(state)
        state.bitmaps[(state._counter - 1) % 2] ^= action_bit
        state._height[action] -= 1
        state._action_log.pop(-1)
        state._counter -= 1
        return state

    def is_win(self, state, player):
        return bool(x_connected(state.bitmaps[player], self.x))

    def is_end(self, state):
        return not self.valid_actions(state)

    @property
    def initial(self):
        return self._STATE_CLS(
            [0, 0], [], {col: col * 7 for col in range(self.columns)}, 0
        )

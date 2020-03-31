"""
This module defines the Connect-X game.
"""

from copy import deepcopy

import numpy as np
from kaggle_environments.utils import Struct

from .game import Game, GameState
from connect_x.utils.converter import (
    board_to_bitmaps,
    bitmaps_to_matrix,
    bitmaps_to_board,
)


class ConnectXState(GameState):
    """
    """

    def __init__(self, bitmaps, action_log, height, counter, mark):
        self.bitmaps = bitmaps
        self._action_log = action_log
        self._height = height
        self.counter = counter
        self.mark = mark

    @classmethod
    def from_observation(cls, observation):
        obj = cls(
            board_to_bitmaps(observation.board), [], None, None, observation.mark,
        )
        obj._height = obj._update_height()
        obj.counter = obj._update_counter()
        return obj

    def to_observation(self):
        return Struct(board=bitmaps_to_board(self.bitmaps), mark=self.mark,)

    def _update_height(self):
        additional_height = (bitmaps_to_matrix(self.bitmaps) != 0).sum(axis=0)
        base_height = np.array([i * 7 for i in range(7)])
        return list(base_height + additional_height)

    def _update_counter(self):
        return bin(self.bitmaps[0])[2:].count("1") + bin(self.bitmaps[1])[2:].count("1")

    @property
    def state_hash(self):
        return 2 * self.bitmaps[0] + self.bitmaps[1]

    def __repr__(self):
        attr_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attr_str})"

    def __eq__(self, other):
        return (
            self.bitmaps == other.bitmaps
            and self._action_log == other._action_log
            and self.mark == other.mark
        )


class ConnectXGame(Game):
    """
    """

    _STATE_CLS = ConnectXState

    def __init__(self, rows=6, columns=7, x=4, timeout=None, steps=1000):
        assert rows == 6, "The game only supports rows=6 for now."
        assert columns == 7, "The game only supports columns=7 for now."
        assert x == 4, "The game only supports x=4 for now."
        self.rows = rows
        self.columns = columns
        self.x = 4
        self._timeout = timeout
        self._steps = steps
        self._ACTIONS = range(columns)
        self._TOP = int("_".join("1000000" for _ in range(columns)), 2)

    @classmethod
    def from_configuration(cls, configuration):
        return cls(
            rows=configuration.rows,
            columns=configuration.columns,
            x=configuration.inarow,
            timeout=configuration.timeout,
            steps=configuration.steps,
        )

    def to_configuration(self):
        return Struct(
            rows=self.rows,
            columns=self.columns,
            inarow=self.x,
            timeout=self._timeout,
            steps=self._steps,
        )

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
        state.bitmaps[state.counter % 2] ^= action_bit
        state._height[action] += 1
        state._action_log.append(action)
        state.counter += 1
        return state

    def undo(self, state, inplace=False):
        action = state._action_log[-1]
        action_bit = 1 << (state._height[action] - 1)
        if not inplace:
            state = deepcopy(state)
        state.bitmaps[(state.counter - 1) % 2] ^= action_bit
        state._height[action] -= 1
        state._action_log.pop(-1)
        state.counter -= 1
        return state

    def connected(self, state, player, x):
        """
        Returns how many times x are connected in the state of player.
        """
        directions = [1, 6, 7, 8]

        def _connected(bitmap, direction, x):
            assert x > 0
            return bin(
                np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)])
            )[2:].count("1")

        return np.array(
            [
                _connected(state.bitmaps[player], direction, x)
                for direction in directions
            ]
        ).sum()

    def is_win(self, state, player):
        return bool(self.connected(state, player, self.x))

    def is_draw(self, state):
        return not self.valid_actions(state)

    @property
    def initial(self):
        return self._STATE_CLS(
            [0, 0], [], {col: col * 7 for col in range(self.columns)}, 0, 1
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(rows={self.rows}, columns={self.columns}, x={self.x})"

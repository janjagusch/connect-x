"""
This module defines the Connect-X game.
"""

from copy import deepcopy

import numpy as np
from kaggle_environments.utils import Struct

from connect_x.game.game import Game, GameState
from connect_x.utils.converter import (
    board_to_bitmaps,
    bitmaps_to_matrix,
    bitmaps_to_board,
)


class ConnectXState(GameState):
    """
    This class represent a game state for Connect-X.

    Args:
        bitmaps (list): The board, represented as a list of two integers.
        action_log (list): The log of previously executed actions.
        height (list): The height of each column in the board,
            in bitmap representation.
        counter (int): The number of turn already played.
        mark (int): The mark of the player.

    """

    def __init__(self, bitmaps, action_log=None, height=None, counter=None, mark=None):
        self.bitmaps = bitmaps
        self.action_log = action_log or []
        self.height = height or self._update_height(bitmaps)
        self.counter = counter or self._update_counter(bitmaps)
        self._mark = mark

    @classmethod
    def from_observation(cls, observation, rows, columns):
        """
        Creates a ConnectXState from an observation.

        Args:
            observation (kaggle_environments.utils.Struct): The observation.
            rows (int): The number of rows.
            columns (int): The number of columns.

        Returns:
            ConnectXState: The state.
        """
        assert rows == 6, "The state only supports rows=6 for now."
        assert columns == 7, "The game only supports columns=7 for now."
        return cls(
            bitmaps=board_to_bitmaps(observation.board, rows, columns),
            mark=observation.mark,
        )

    def to_observation(self):
        """
        Creates an observation from the state.

        Returns:
            kaggle_environments.utils.Struct: The observation.
        """
        return Struct(board=bitmaps_to_board(self.bitmaps), mark=self._mark,)

    @staticmethod
    def _update_height(bitmaps):
        additional_height = (bitmaps_to_matrix(bitmaps) != 0).sum(axis=0)
        base_height = np.array([i * 7 for i in range(7)])
        return list(base_height + additional_height)

    @staticmethod
    def _update_counter(bitmaps):
        return bin(bitmaps[0])[2:].count("1") + bin(bitmaps[1])[2:].count("1")

    def __hash__(self):
        return int(self.bitmaps[0] + (self.bitmaps[0] | self.bitmaps[1]))

    def __repr__(self):
        attr_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attr_str})"

    def __eq__(self, other):
        return self.bitmaps == other.bitmaps


class ConnectXGame(Game):
    """
    This class represents the Connect-X game.

    Args:
        rows (int): The number of rows.
        columns (int): The number of columns.
        x (int): The number of tokens connected to win.
        timeout (int): The timeout for the turn.
        steps (int): The maximum number of steps.
    """

    _STATE_CLS = ConnectXState

    def __init__(self, rows=6, columns=7, x=4, timeout=None, steps=1000):
        assert rows == 6, "The game only supports rows=6 for now."
        assert columns == 7, "The game only supports columns=7 for now."
        assert x == 4, "The game only supports x=4 for now."
        self.rows = rows
        self.columns = columns
        self.x = 4
        self.timeout = timeout
        self._steps = steps
        self._actions = range(columns)
        self._top = int("_".join("1000000" for _ in range(columns)), 2)

    @classmethod
    def from_configuration(cls, configuration):
        """
        Creates a game from a configuration.

        Args:
            configuration (kaggle_environments.utils.Struct): The configuration.

        Returns:
            ConnectXGame: The game.
        """
        return cls(
            rows=configuration.rows,
            columns=configuration.columns,
            x=configuration.inarow,
            timeout=configuration.timeout,
            steps=configuration.steps,
        )

    def to_configuration(self):
        """
        Creates a configuration from a game.

        Returns:
            kaggle_environments.utils.Struct: The configuration.
        """
        return Struct(
            rows=self.rows,
            columns=self.columns,
            inarow=self.x,
            timeout=self.timeout,
            steps=self._steps,
        )

    def valid_actions(self, state):
        return [
            action
            for action in self._actions
            if not self._top & (1 << state.height[action])
        ]

    def do(self, state, action, inplace=False):
        action_bit = 1 << state.height[action]
        if not inplace:
            state = deepcopy(state)
        state.bitmaps[state.counter % 2] ^= action_bit
        state.height[action] += 1
        state.action_log.append(action)
        state.counter += 1
        return state

    def undo(self, state, inplace=False):
        action = state.action_log[-1]
        action_bit = 1 << (state.height[action] - 1)
        if not inplace:
            state = deepcopy(state)
        state.bitmaps[(state.counter - 1) % 2] ^= action_bit
        state.height[action] -= 1
        state.action_log.pop(-1)
        state.counter -= 1
        return state

    def is_win(self, state, player):
        directions = [1, 6, 7, 8]

        # pylint: disable=no-member
        def _is_win(bitmap, direction, x):
            return np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)])

        # pylint: enable=no-member

        return any(
            _is_win(state.bitmaps[player], direction, self.x)
            for direction in directions
        )

    def is_draw(self, state):
        return not self.valid_actions(state)

    @property
    def initial(self):
        return self._STATE_CLS(bitmaps=[0, 0], mark=1,)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rows={self.rows}, "
            f"columns={self.columns}, x={self.x})"
        )

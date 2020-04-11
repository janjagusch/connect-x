"""
This module implements the Minimax algorithm.
"""

import asyncio

import numpy as np

from connect_x.utils.logger import setup_logger
from connect_x.utils.async_lru import alru_cache


_LOGGER = setup_logger(__name__)


class Minimax:
    """
    This class defines the Minimax algorithm.

    Args:
        game (connect_x.game.game.Game): The game.
        player (int): The player.
        depth (int): The maximum depth for the search tree.
        heuristic_func (callable, optional): The function that evaluates non-terminal
            nodes at maximum depth. Takes two arguments: `state` and `player`.
        order_actions_func (callable, optional): The function that defines in which
            order the valid actions should be evaluated. Takes one argument: `actions`.
        alpha_beta_pruning (bool, optional): Whether alpha-beta pruning should be
            applied to the search tree.
        inplace (bool, optional): Whether the initial game state should be evaluated
            inplace.
    """

    def __init__(
        self,
        game,
        player,
        depth,
        heuristic_func=None,
        order_actions_func=None,
        alpha_beta_pruning=True,
        inplace=False,
    ):
        self.game = game
        self.player = player
        self.depth = depth
        self.heuristic_func = heuristic_func or (lambda state, player: 0)
        self.order_actions_func = order_actions_func or (lambda actions: actions)
        self.alpha_beta_pruning = alpha_beta_pruning
        self.inplace = inplace

    def _is_terminated(self, state):
        if self.game.is_win(state, self.player):
            return np.inf, True
        if self.game.is_win(state, 1 - self.player):
            return -np.inf, True
        if self.game.is_draw(state):
            return 0, True
        return None, False

    def _static_evaluation(self, *, state, depth):
        value, terminated = self._is_terminated(state)
        if terminated:
            return value
        if not depth:
            return self.heuristic_func(state, self.player)
        return None

    def _child_states(self, state, actions):
        for action in actions:
            try:
                yield action, self.game.do(state, action, inplace=self.inplace)
            finally:
                if self.inplace:
                    self.game.undo(state, inplace=self.inplace)

    @alru_cache(maxsize=100000)
    async def _minimax(self, state, depth, alpha, beta, maximize):
        await asyncio.sleep(0)
        value = self._static_evaluation(state=state, depth=depth)
        if value is not None:
            return value, None

        actions = self.game.valid_actions(state)
        actions = self.order_actions_func(actions)

        best_value = None
        best_action = None

        if maximize:
            for action, child in self._child_states(state, actions):
                value, _ = await self._minimax(
                    child, depth - 1, alpha, beta, maximize=False
                )
                if best_value is None or best_value < value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)
                if self.alpha_beta_pruning and beta <= alpha:
                    break

        else:
            for action, child in self._child_states(state, actions):
                value, _ = await self._minimax(
                    child, depth - 1, alpha, beta, maximize=True
                )
                if best_value is None or best_value > value:
                    best_value = value
                    best_action = action
                beta = min(beta, value)
                if self.alpha_beta_pruning and beta <= alpha:
                    break

        return best_value, best_action

    async def __call__(self, state):

        self._minimax.cache_clear()

        meta_state = await self._minimax(
            state=state, depth=self.depth, alpha=-np.inf, beta=np.inf, maximize=True,
        )

        # pylint: disable=no-value-for-parameter
        _LOGGER.debug(self._minimax.cache_info())
        # pylint: enable=no-value-for-parameter
        _LOGGER.debug(meta_state)

        return meta_state

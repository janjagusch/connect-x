"""
This module implements the Minimax search algorithm with Alpha-Beta pruning.
The implementation is taken from here:
https://github.com/aimacode/aima-python/blob/master/games.py
"""

import functools

import numpy as np

import asyncio

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


class StateValueCache:
    """
    A cache that stores the value for every state.
    """

    def __init__(self, func, cache=None):
        functools.update_wrapper(self, func)
        self.func = func
        self.cache = cache or {}
        self.cache_calls = 0

    def _cached_value(self, key):
        if key is not None:
            value = self.cache.get(key)
            if value is not None:
                self.cache_calls += 1
            return value
        return None

    def _cache_value(self, key, value):
        if key is not None:
            self.cache[key] = value

    def reset_cache(self):
        """
        Resets the cache.
        """
        self.cache = {}

    async def __call__(self, *args, **kwargs):
        key = kwargs.get("state").state_hash
        value = self._cached_value(key) or await self.func(*args, **kwargs)
        self._cache_value(key, value)
        return value


def is_terminated(state, game, player):
    """
    Returns the value of the game state and whether the game has terminated.

    Args:
        state (connect_x.game.connect_x.ConnectXState): The state.
        game (connect_x.game.connect_x.ConnectXGame): The game.
        player (int): The player (1 or 0).

    Returns:
        tuple: The value of the state for the player and whether the game has ended.
    """
    if game.is_win(state, player):
        return np.inf, True
    if game.is_win(state, 1 - player):
        return -np.inf, True
    if game.is_draw(state):
        return 0, True
    return None, False


async def negamax(
    game, state, depth, player, heuristic_func, order_actions_func=None,
):
    """
    Applies the Negamax algorithm to the game to determine the next best action for
    the player, given the state.

    Args:
        game (connect_x.game.connect_x.ConnectXGame): The game.
        state (connect_x.game.connect_x.ConnectXState): The state.
        depth (int): The maximum depth of the Negamax tree.
        player (int): The player (1 or 0).
        heuristic_func (callable, optional): The heuristic function for the state when
            the tree can not be resolved up to a terminal leaf.
        order_actions_func (callable, optional): The function that determines in which
            order the actions will be evaluated.

    Returns:
        float: When `return_cache=False`.
        dict: When `return_cache=True`.
    """
    heuristic_func = heuristic_func or (lambda state, player: 0)
    order_actions_func = order_actions_func or (lambda actions: actions)
    alpha = -np.inf
    beta = np.inf

    @StateValueCache
    async def _negamax(state, game, depth, alpha, beta, maximize):
        value, terminated = is_terminated(state, game, player)
        if terminated:
            return value * maximize
        if depth == 0:
            return heuristic_func(state, player) * maximize

        actions = game.valid_actions(state)
        actions = order_actions_func(actions)

        value = -np.inf

        for action in actions:
            await asyncio.sleep(0)
            v = await _negamax(
                state=game.do(state, action),
                game=game,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                maximize=-maximize,
            )
            value = max(value, -v)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value

    _negamax.reset_cache()
    _ = await _negamax(
        state=state, game=game, depth=depth, alpha=alpha, beta=beta, maximize=1
    )

    return _best_action(game, state, _negamax.cache, order_actions_func)


def _best_action(game, state, cache, order_actions_func):
    valid_actions = order_actions_func(game.valid_actions(state))
    valid_states = [game.do(state, action) for action in valid_actions]
    values = [cache.get(state.state_hash, np.inf) * -1 for state in valid_states]
    _LOGGER.debug(list(zip(valid_actions, values)))
    return valid_actions[np.argmax(values)]

"""
This module implements the Minimax search algorithm with Alpha-Beta pruning.
The implementation is taken from here: https://github.com/aimacode/aima-python/blob/master/games.py
"""

import functools


class StateValueCache:
    """
    A cache that stores the value for every state.
    """
    
    def __init__(self, func, cache=None):
        functools.update_wrapper(self, func)
        self.func = func
        self.cache = cache or {}
        
    def _cached_value(self, key):
        if key is not None:
            return self.cache.get(key)
        return None

    def _cache_value(self, key, value):
        if key is not None:
            self.cache[key] = value
        
    def __call__(self, *args, **kwargs):
        key = kwargs.get("state").state_hash
        value = self._cached_value(key) or self.func(*args, **kwargs)
        self._cache_value(key, value)
        return value


def minimax(state, game, max_depth=4, maximize=True, eval_func=None, order_actions_func=None)



def minimax(state, game, max_depth=4, eval_func=None, order_actions_func=None, maximize=True):

    player = game.to_move(state)

    @StateValueCache
    def max_value(state, alpha, beta, depth):
        value, terminated = eval_func(state)
        if terminated or depth > max_depth:
            return value
        value = -np.inf
        for action in order_actions_func(game.actions(state)):
            value = max(
                value, min_value(game.result(state, action), alpha, beta, depth + 1)
            )
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    @StateValueCache
    def min_value(state, alpha, beta, depth):

        value, terminated = eval_func(state)
        if terminated or depth > max_depth:
            return value
        value = np.inf
        for action in order_actions_func(game.actions(state)):
            value = min(
                value, max_value(game.result(state, action), alpha, beta, depth + 1)
            )
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    # Body of alpha_beta_cutoff_search starts here:
    eval_func = eval_func or (lambda state: game.utility(state, player))
    order_actions_func = order_actions_func or (lambda actions: actions)
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for action in order_actions_func(game.actions(state)):
        value = min_value(game.result(state, action), best_score, beta, 1)
        if value > best_score:
            best_score = value
            best_action = action
    return best_action

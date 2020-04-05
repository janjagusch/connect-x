"""
This module implements the Minimax search algorithm with Alpha-Beta pruning.
The implementation is taken from here:
https://github.com/aimacode/aima-python/blob/master/games.py
"""

import numpy as np

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


def is_terminated(game, state, player):
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


def negamax(
    game,
    state,
    player,
    depth,
    heuristic_func=None,
    order_actions_func=None,
    alpha_beta_pruning=True,
    inplace=False,
    maximize=1,
):
    """
    Executes the negamax algorithm.

    Args:
        game (connect_x.game.Game): The 2-player game.
        state (connect_x.game.GameState): The game state.
        player (int): The player.
        depth (int): The maximum tree depth.
        heuristic_func (callable, optional): The evaluation function for states at
            maximum depth that are not terminal leaves.
        order_actions_func (callable, optional): The function that determines in which
            order to evaluate actions.
        alpha_beta_pruning (bool, optional): Whether to apply alpha-beta pruning.
        inplace (bool, optional): Whether to transform states inplace.
            Consumes less memory.
        maximize (bool, optional): Whether to start with a maximize (1) or a minimize
            (-1) round.

    Returns:
        int: The best next action.
    """

    def _negamax(state, depth, alpha, beta, maximize):
        value, terminated = is_terminated(game, state, player)
        if terminated:
            return value * maximize, None
        if depth == 0:
            return heuristic_func(state, player) * maximize, None

        actions = game.valid_actions(state)
        actions = order_actions_func(actions)

        best_value = None
        best_action = None

        for action in actions:
            new_state = game.do(state, action, inplace=inplace)
            new_value, _ = _negamax(
                state=new_state,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                maximize=-maximize,
            )
            new_value *= -1

            if best_value is None or new_value > best_value:
                best_value = new_value
                best_action = action
                alpha = new_value

            if inplace:
                state = game.undo(new_state, inplace=inplace)

            if alpha_beta_pruning and alpha >= beta:
                break

        return best_value, best_action

    heuristic_func = heuristic_func or (lambda state, player: 0)
    order_actions_func = order_actions_func or (lambda actions: actions)

    alpha = -np.inf
    beta = np.inf

    value, action = _negamax(
        state=state, depth=depth, alpha=alpha, beta=beta, maximize=maximize
    )

    _LOGGER.debug(f"Best action '{action}' with value '{value}'.")

    return action

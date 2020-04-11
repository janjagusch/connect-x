"""
This module contains the submission for the Kaggle competition.
Version: 0.6.0.
"""

import asyncio

from datetime import datetime

from connect_x.game import ConnectXGame, ConnectXState
from connect_x.action_catalogue import get_action
from connect_x.agents import negamax
from connect_x.config import heuristic, order_actions, TIMEOUT_BUFFER

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


def _catalogued_action(state, player):
    return get_action(state, player)


def _planned_action(game, state, player):
    result = None
    timeout = game.timeout * TIMEOUT_BUFFER
    min_depth = 0
    max_depth = game.rows * game.columns - state.counter
    _LOGGER.debug(f"Setting internal timeout: {timeout}.")

    async def iterative_deepening():
        """
        Repeats the minimax algorithm with increasing depth, and
        saves the latest result to a nonlocal variable in the closure.
        """
        nonlocal result
        for depth in range(min_depth, max_depth + 1):
            _LOGGER.debug(f"Minimax depth: {depth}.")
            result = await negamax(
                game=game,
                state=state,
                player=player,
                depth=depth,
                heuristic_func=heuristic,
                order_actions_func=order_actions,
            )

    async def call_with_timeout(afun, timeout=timeout):
        """
        Calls an async function with a timeout. Note that this requires that
        the function itself is async and runs on the event loop, such that it
        can be interrupted.
        """
        try:
            await asyncio.wait_for(afun(), timeout=timeout)
        except asyncio.TimeoutError:
            _LOGGER.debug(f"Timed out internally")
            return

    asyncio.run(call_with_timeout(iterative_deepening))
    return result


def act(observation, configuration):
    """
    Decides what action to do next.

    Args:
        observation (kaggle_environments.utils.Struct): The observation.
        configuration (kaggle_environments.utils.Struct): The configuration.

    Returns:
        int: The action.
    """

    start = datetime.now()

    game = ConnectXGame.from_configuration(configuration)
    state = ConnectXState.from_observation(
        observation, configuration.rows, configuration.columns
    )
    player = observation.mark - 1

    _LOGGER.info(f"Round #{state.counter}!")
    _LOGGER.debug(f"State hash: '{state.state_hash}'")
    _LOGGER.debug(f"Player: '{player}'.")
    action = _catalogued_action(state, player) or _planned_action(game, state, player)
    if _catalogued_action(state, player):
        _LOGGER.debug(f"Cache hit")
    end = datetime.now()
    _LOGGER.info(f"Action selected: '{action}'.")
    _LOGGER.debug(f"Time taken: {end - start}.")

    return int(action)

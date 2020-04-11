"""
This module contains the submission for the Kaggle competition.
"""
from datetime import datetime

from connect_x.game import ConnectXGame, ConnectXState
from connect_x.action_catalogue import get_action
from connect_x.agents import Minimax, IterativeDeepening
from connect_x.config import heuristic, order_actions, TIMEOUT_BUFFER, INPLACE

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


def _catalogued_action(state, player):
    return get_action(state, player)


def _planned_action(game, state, player):
    return IterativeDeepening(
        lambda state, depth: Minimax(
            game,
            player=player,
            depth=depth,
            heuristic_func=heuristic,
            order_actions_func=order_actions,
            inplace=INPLACE,
        )(state),
        timeout=TIMEOUT_BUFFER,
        min_depth=1,
        max_depth=game.rows * game.columns - state.counter,
    )(state)


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
    _LOGGER.debug(f"Player: '{player}'.")
    action = _catalogued_action(state, player) or _planned_action(game, state, player)
    end = datetime.now()
    _LOGGER.info(f"Action selected: '{action}'.")
    _LOGGER.debug(f"Time taken: {end - start}.")

    return int(action)

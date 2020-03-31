"""
This module contains the submissoin for the Kaggle competition.
"""

from datetime import datetime

import numpy as np

from connect_x.game import ConnectXGame, ConnectXState
from connect_x.action_catalogue import get_action
from connect_x.minimax import negamax
from connect_x.config import heuristic, order_actions, DEPTH

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


def _catalogued_action(state):
    return get_action(state)


def _planned_action(game, state):
    cache = negamax(
        game=game,
        state=state,
        depth=DEPTH,
        heuristic_func=heuristic,
        order_actions_func=order_actions,
        player=state.mark - 1,
        return_cache=True,
    )
    valid_actions = order_actions(game.valid_actions(state))
    valid_states = [game.do(state, action) for action in valid_actions]
    values = [cache.cache.get(state.state_hash, np.inf) * -1 for state in valid_states]
    _LOGGER.debug({action: value for action, value in zip(valid_actions, values)})
    return valid_actions[np.argmax(values)]


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
    state = ConnectXState.from_observation(observation)
    action = _catalogued_action(state) or _planned_action(game, state)
    end = datetime.now()
    _LOGGER.info(f"Action selected: '{action}'.")
    _LOGGER.debug(f"Time taken: {end - start}.")

    return int(action)

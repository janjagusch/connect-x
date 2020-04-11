"""
This module provides tests for the `connect_x.game.connect_x` module.
"""

import pytest
from kaggle_environments.utils import Struct

from connect_x.game.connect_x import ConnectXState, ConnectXGame
from connect_x.utils.converter import bitmaps_to_board


# pylint: disable=protected-access


class TestConnectXState:
    """
    Tests the ConnectXState class.
    """

    @staticmethod
    def test_from_observation(observation, configuration):
        state = ConnectXState.from_observation(
            observation, configuration.rows, configuration.columns
        )
        assert isinstance(state, ConnectXState)
        assert state._mark == observation.mark
        assert bitmaps_to_board(state.bitmaps) == observation.board

    @pytest.fixture(name="state")
    @staticmethod
    def state(observation, configuration):
        """
        Initial state from observation.
        """
        return ConnectXState.from_observation(
            observation, configuration.rows, configuration.columns
        )

    @staticmethod
    def test_to_observation(state, observation):
        obs = state.to_observation()
        assert isinstance(obs, Struct)
        assert obs == observation

    @staticmethod
    def test__update_height(state):
        assert state.height == state._update_height(state.bitmaps)

    @staticmethod
    def test__update_counter(state):
        assert state.counter == state._update_counter(state.bitmaps)

    @staticmethod
    def test___repr__(state):
        assert isinstance(state.__repr__(), str)


class TestConnectXGame:
    """
    Tests the ConnectXGame class.
    """

    @staticmethod
    def test_from_configuration(configuration):
        game = ConnectXGame.from_configuration(configuration)
        assert isinstance(game, ConnectXGame)
        assert game.rows == configuration.rows
        assert game.columns == configuration.columns
        assert game.x == configuration.inarow

    @pytest.fixture(name="game")
    @staticmethod
    def game_(configuration):
        """
        Game from configuration.
        """
        return ConnectXGame.from_configuration(configuration)

    @pytest.fixture(name="state", scope="function")
    @staticmethod
    def state_(game):
        """
        Initial state from game.
        """
        return game.initial

    @staticmethod
    def test_to_configuration(game, configuration):
        cfg = game.to_configuration()
        assert isinstance(cfg, Struct)
        assert cfg == configuration

    @staticmethod
    def test_valid_actions(game, state):
        valid_actions = game.valid_actions(state)
        assert isinstance(valid_actions, list)

    @staticmethod
    def test_do(game, state):
        new_state = game.do(state, 0)
        assert isinstance(state, ConnectXState)
        assert new_state.action_log[-1] == 0
        assert new_state != state
        assert new_state is not state

    @staticmethod
    def test_do_inplace(game, state):
        new_state = game.do(state, 0, inplace=True)
        assert isinstance(state, ConnectXState)
        assert new_state is state

    @staticmethod
    def test_undo(game, state):
        new_state = game.do(state, 0)
        old_state = game.undo(new_state)
        assert old_state.action_log == []
        assert old_state == state
        assert old_state is not state

    @staticmethod
    def test_is_win(game, state):
        assert not game.is_win(state, 0)
        assert not game.is_win(state, 1)

    @staticmethod
    def test_is_draw(game, state):
        assert not game.is_draw(state)

    @staticmethod
    def test_initial(game):
        initial = game.initial
        assert isinstance(initial, ConnectXState)

    @staticmethod
    def test___repr__(game):
        assert isinstance(game.__repr__(), str)

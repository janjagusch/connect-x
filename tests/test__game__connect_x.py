"""
This module provides tests for the `connect_x.game.connect_x` module.
"""

import pytest
from kaggle_environments.utils import Struct

from connect_x.game.connect_x import ConnectXState, ConnectXGame
from connect_x.utils.converter import bitmaps_to_board


class TestConnectXState:
    """
    Tests the ConnectXState class.
    """

    def test_from_observation(self, observation):
        state = ConnectXState.from_observation(observation)
        assert isinstance(state, ConnectXState)
        assert state.mark == observation.mark
        assert bitmaps_to_board(state.bitmaps) == observation.board

    @pytest.fixture(name="state")
    @staticmethod
    def state(observation):
        return ConnectXState.from_observation(observation)

    def test_to_observation(self, state, observation):
        obs = state.to_observation()
        assert isinstance(obs, Struct)
        assert obs == observation

    def test__update_height(self, state):
        assert state._height == state._update_height()

    def test__update_counter(self, state):
        assert state.counter == state._update_counter()

    def test_state_hash(self, state):
        assert isinstance(state.state_hash, int)

    def test___repr__(self, state):
        assert isinstance(state.__repr__(), str)


class TestConnectXGame:
    """
    Tests the ConnectXGame class.
    """

    def test_from_configuration(self, configuration):
        game = ConnectXGame.from_configuration(configuration)
        assert isinstance(game, ConnectXGame)
        assert game.rows == configuration.rows
        assert game.columns == configuration.columns
        assert game.x == configuration.inarow

    @pytest.fixture(name="game")
    @staticmethod
    def game_(configuration):
        return ConnectXGame.from_configuration(configuration)

    @pytest.fixture(name="state", scope="function")
    @staticmethod
    def state_(game):
        return game.initial

    def test_to_configuration(self, game, configuration):
        cfg = game.to_configuration()
        assert isinstance(cfg, Struct)
        assert cfg == configuration

    def test_valid_actions(self, game, state):
        valid_actions = game.valid_actions(state)
        assert isinstance(valid_actions, list)

    def test_do(self, game, state):
        new_state = game.do(state, 0)
        assert isinstance(state, ConnectXState)
        assert new_state._action_log[-1] == 0
        assert new_state != state
        assert new_state is not state

    def test_do_inplace(self, game, state):
        new_state = game.do(state, 0, inplace=True)
        assert isinstance(state, ConnectXState)
        assert new_state is state

    def test_undo(self, game, state):
        new_state = game.do(state, 0)
        old_state = game.undo(new_state)
        assert old_state._action_log == []
        assert old_state == state
        assert old_state is not state

    def test_connected(self, game, state):
        assert not game.connected(state, 0, 1)
        assert not game.connected(state, 1, 1)

    def test_is_win(self, game, state):
        assert not game.is_win(state, 0)
        assert not game.is_win(state, 1)

    def test_is_draw(self, game, state):
        assert not game.is_draw(state)

    def test_initial(self, game):
        initial = game.initial
        assert isinstance(initial, ConnectXState)

    def test___repr__(self, game):
        assert isinstance(game.__repr__(), str)

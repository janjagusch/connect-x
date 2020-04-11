"""
This module represents a generic game and game state.
"""


class GameState:
    """
    This class represent a abitrary state for a game.
    """


class Game:
    """
    This class represent a abstract game.
    """

    _STATE_CLS = GameState

    def valid_actions(self, state):
        """
        Return a list of the allowable moves at this point.
        """
        raise NotImplementedError

    # pylint: disable=invalid-name
    def do(self, state, action, inplace=False):
        """
        Return the state that results from making an action from a state.

        Args:
            state (GameState): The state.
            action (int): The action.
            inplace (boolean): If `False` returns a new game state. If `True`
                overwrites the initial game state.

        Returns:
            GameState: The game state after performing the action.
        """
        raise NotImplementedError

    # pylint: enable=invalid-name

    def undo(self, state, inplace=False):
        """
        Returns the state that results from undoing the last action from a state.

        Args:
            state (GameState): The state.
            inplace (boolean): If `False` returns a new game state. If `True`
                overwrites the initial game state.

        Returns:
            GameState: The game state before performing the last action.
        """
        raise NotImplementedError

    @property
    def initial(self):
        """
        Returns the inital game state for this game.

        Returns:
            GameState: The initial game state.
        """
        raise NotImplementedError

    def is_win(self, state, player):
        """
        Returns `True` if player has won in the state.

        Args:
            state (GameState): The state.
            player (int): The player.

        Returns:
            boolean: Whether player has won in the state.
        """
        raise NotImplementedError

    def is_draw(self, state):
        """
        Returns `True` if the game has ended in a draw.

        Args:
            state (GameState): The game state.

        Returns:
            boolean: Whether the game has ended in a draw.
        """
        raise NotImplementedError

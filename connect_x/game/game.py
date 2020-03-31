"""
"""


class GameState:
    """
    """

    @property
    def state_hash(self):
        return None


class Game:
    """
    """

    _STATE_CLS = GameState

    def valid_actions(self, state):
        """
        Return a list of the allowable moves at this point.
        """
        raise NotImplementedError

    def do(self, state, action, inplace=False):
        """Return the state that results from making an action from a state."""
        raise NotImplementedError

    def undo(self, state, inplace=False):
        """
        """
        raise NotImplementedError

    @classmethod
    def initial(cls):
        """
        """
        raise NotImplementedError

    def is_win(self, state, player):
        """
        """
        raise NotImplementedError

    def is_draw(self, state):
        """
        Warning! This does not check for `is_win`.
        """
        raise NotImplementedError

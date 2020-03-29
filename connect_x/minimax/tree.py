"""
This module provides tree structures for the Minimax algorithm.
"""

from connect_x.actions import possible_actions, step
from connect_x.evaluate import evaluate
from connect_x.utils.board import (
    other_token,
    TOKEN_ME,
)


class TreeNode:
    """
    A generic tree node.

    Args:
        parent (TreeNode): The parent node.
        children (list): The child nodes.
    """

    def __init__(self, parent=None, children=None):
        self.parent = parent
        self.children = children or []
        if self.parent:
            self.parent.children.append(self)


class ConnectXNode(TreeNode):
    """
    A tree node for Connect X.

    Args:
        matrix (np.array): The board state as matrix.
        configuration (kaggle_environments.utils.Struct): The configuration.
        action (int): The index of the column where you inserted the token.
        next_token (int): The next token (mark independent).
        parent (ConnectXNode): The parent node.
        children (list): The child nodes.
    """

    def __init__(
        self,
        matrix,
        configuration,
        action=None,
        next_token=None,
        parent=None,
        children=None,
    ):
        super().__init__(parent, children)
        self.matrix = matrix
        self.configuration = configuration
        self.action = action
        self.next_token = next_token

    @property
    def possible_actions(self):
        """
        Returns a list of possible actions you can take from this node.

        Returns:
            list: List of possible actions.
        """
        return possible_actions(self.matrix)

    def step(self, action):
        """
        Executes an action and returns the new child node.

        Args:
            action (int): The index of the column where you want to insert the token.

        Returns:
            ConnectXNode: The new child node.
        """
        next_token = self.next_token or TOKEN_ME
        return self.__class__(
            matrix=step(self.matrix, action, next_token),
            configuration=self.configuration,
            action=action,
            next_token=other_token(next_token),
            parent=self,
        )

    def make_children(self):
        """
        Generates all child nodes.

        Returns:
            list: A list of child nodes.
        """
        self.children = [self.step(action) for action in self.possible_actions]
        return self.children

    @property
    def value(self):
        """
        Calculates the value of the node.
        """
        return evaluate(self.matrix, self.configuration)

"""
This module provides tree structures for the Minimax algorithm.
"""

from connect_x.actions import possible_actions, step
from connect_x.evaluate import evaluate
from connect_x.utils import other_mark


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
        observation (kaggle_environments.utils.Struct): The observation.
        configuration (kaggle_environments.utils.Struct): The configuration.
        action (int): The index of the column where you inserted the token.
        mark (int): The mark of the token.
        parent (TreeNode): The parent node.
        children (list): The child nodes.
    """

    def __init__(
        self,
        observation,
        configuration,
        action=None,
        mark=None,
        parent=None,
        children=None,
    ):
        super().__init__(parent, children)
        self.observation = observation
        self.configuration = configuration
        self.action = action
        self.mark = mark

    @property
    def possible_actions(self):
        """
        Returns a list of possible actions you can take from this node.

        Returns:
            list: List of possible actions.
        """
        return possible_actions(self.observation, self.configuration)

    def step(self, action):
        """
        Executes an action and returns the new child node.

        Args:
            action (int): The index of the column where you want to insert the token.

        Returns:
            ConnectXNode: The new child node.
        """
        mark = other_mark(self.mark) or self.observation.mark
        return self.__class__(
            observation=step(self.observation, self.configuration, action, mark),
            configuration=self.configuration,
            action=action,
            parent=self,
            mark=mark,
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
        return evaluate(self.observation, self.configuration)

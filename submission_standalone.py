from copy import copy

import numpy as np


def other_mark(mark):
    """
    Given the mark of a token, returns the other mark.

    Args:
        mark (int): The mark of the token.

    Returns:
        int: The other mark or `None`, when mark is `None`.
    """
    if not mark:
        return None
    assert mark in (1, 2)
    return 2 if mark == 1 else 1


def board_to_matrix(board, n_rows, n_cols):
    """
    Converts a board into a numpy matrix.

    Args:
        board (list): The board state.
        n_rows (int): Number of rows on the board.
        n_cols (int): Number of columns on the board.

    Returns:
        np.array: The board as a matrix.
    """
    return np.array(board).reshape(n_rows, n_cols)


def matrix_to_board(matrix):
    """
    Converts a matrix into a board.

    Args:
        matrix (np.array): The board matrix.

    Returns:
        list: The board as a list.
    """
    return matrix.reshape(1, -1).tolist()[0]


def rolling_window(array, window_size):
    """
    Returns rolling windows over a 1-dimensional array.

    Args:
        array (np.array): A 1-dimensional arary.
        window_size (int): The window size.

    Returns:
        list: List of np.array objects.
    """
    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def _diagonals(matrix):
    return [matrix.diagonal(i) for i in range(-matrix.shape[0] + 1, matrix.shape[1])]


def matrix_diagonals(matrix):
    """
    Returns all diagonals of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return _diagonals(matrix) + _diagonals(matrix[::-1])


def matrix_rows(matrix):
    """
    Returns all rows of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return list(matrix)


def matrix_columns(matrix):
    """
    Returns all columns of a 2-dimensional matrix.

    Args:
        matrix (np.array): A 2-dimensional matrix.

    Returns:
        list: List of np.array objects.
    """
    return list(matrix.T)


def _windows(matrix, window_size):
    """
    Calculates all windows that are relevant to evaluate to board state from a matrix.

    Args:
        matrix (np.array): A board matrix.
        window_size (int): The number of token you need to have 'in a row'.

    Returns:
        np.array: The windows of the board.
    """
    windows = []
    # pylint: disable=bad-continuation
    for array in (
        matrix_rows(matrix) + matrix_columns(matrix) + matrix_diagonals(matrix)
    ):
        # pylint: enable=bad-continuation
        if len(array) >= window_size:
            windows.extend(rolling_window(array, window_size))
    return np.array(windows)


def _eval_windows(windows, mark):
    """
    Calculates the evaluation windows, depending on the mark of the token.

    Args:
        windows (np.array): Array of windows.
        mark (int): `1` or `2`.

    Returns:
        np.array: Array of evaluation windows.
    """
    mark_opponent = 2 if mark == 1 else 1
    eval_windows = np.zeros(windows.shape)
    eval_windows[windows == mark] = 1
    eval_windows[windows == mark_opponent] = -1
    return eval_windows


def _evaluate_victory(eval_windows):
    """
    Checks whether evaluation windows contain a victory.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: Whether evaluation windows contain victory.
    """
    return (eval_windows.mean(axis=1) == 1).any()


def _evaluate_board_full(eval_windows):
    """
    Checks whether the board is full.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: Whether the board is full.
    """

    return not (eval_windows == 0).any()


def _evaluate_heuristic(eval_windows):
    """
    Evaluates the board.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        bool: The value of the board.
    """
    values = np.exp2(eval_windows.sum(axis=1))
    not_contains_other = eval_windows.min(axis=1) != -1
    return (values * not_contains_other).mean()


def _evaluate(eval_windows):
    """
    Evaluates the board. Calculates a value for the board and checks whether the game
    has ended.

    Args:
        eval_windows (np.array): Array of evaluation windows.

    Returns:
        tuple: (The value of the board, Whether the game has ended).
    """
    if _evaluate_victory(eval_windows):
        return float("inf"), True
    if _evaluate_board_full(eval_windows):
        return float(0), True
    return _evaluate_heuristic(eval_windows), False


def evaluate(observation, configuration):
    """
    Evaluates an observation.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.

    Returns:
        tuple: (The value of the board, Whether the game has ended).
    """
    mark = observation.mark
    mark_opponent = 2 if mark == 1 else 1

    matrix = board_to_matrix(
        observation.board, configuration.rows, configuration.columns
    )
    windows = _windows(matrix, configuration.inarow)
    eval_windows = _eval_windows(windows, mark)
    eval_windows_opponent = _eval_windows(windows, mark_opponent)
    value, done = _evaluate(eval_windows)
    value_opponent, done_opponent = _evaluate(eval_windows_opponent)
    return value - value_opponent, any([done, done_opponent])


def _possible_actions(matrix):
    """
    Returns all possible actions you can take from a matrix state.

    Args:
        matrix (np.array): The board matrix.

    Returns:
        np.array: The possible actions.
    """
    filter_ = [(array == 0).any() for array in matrix_columns(matrix)]
    return np.arange(matrix.shape[1])[filter_]


def _step(matrix, action, mark):
    """
    Applies an action with a mark to a matrix.

    Args:
        matrix (np.array): The board matrix.
        action (int): The column index where the token should be placed.
        mark (int): The mark of the token.

    Returns:
        np.array: The new token matrix.
    """
    col = matrix[:, action]
    row = np.argwhere(col == 0).max()
    new_matrix = matrix.copy()
    new_matrix[row, action] = mark
    return new_matrix


def possible_actions(observation, configuration):
    """
    Lists all possible actions that can be taken.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.

    Returns:
        list: List of possible actions.
    """
    board = observation.board
    n_rows = configuration.rows
    n_cols = configuration.columns

    matrix = board_to_matrix(board, n_rows, n_cols)
    return list(_possible_actions(matrix))


def step(observation, configuration, action, mark):
    """
    Executes an action and returns the new observation.

    Args:
        observation (dict): The observation.
        configuration (dict): The configuration.
        action (int): The index of the column where you want to insert the token.
        mark (int): The mark of the token.

    Returns:
        dict: The new observation.
    """
    board = observation.board
    n_rows = configuration.rows
    n_cols = configuration.columns

    matrix = board_to_matrix(board, n_rows, n_cols)
    new_matrix = _step(matrix, action, mark)

    new_board = matrix_to_board(new_matrix)

    new_observation = copy(observation)
    new_observation.board = new_board

    return new_observation


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


def minimax(node, max_depth=4, maximize=True, current_depth=None):
    """
    Executes the Minimax algorithm.

    Args:
        node (connect_x.minimax.tree.ConnectXNode): The root node.
        max_depth (int, optional): The maximum recursion depth.
        maximize (bool, optional): Whether to maximize or minimize.
        current_depth (int, optional): The current depth.

    Returns:
        tuple: (Next node to go for, Value of the next node).
    """
    current_depth = current_depth or 0
    value, terminated = node.value
    if (current_depth == max_depth) or terminated:
        return value
    children = node.make_children()
    values = [
        minimax(child, max_depth, not maximize, current_depth + 1) for child in children
    ]
    if maximize:
        value = np.max(values)
        index = np.argmax(values)
    else:
        value = np.min(values)
        index = np.argmin(values)
    if not current_depth:
        return children[index], value
    return value


def act(observation, configuration):
    node = ConnectXNode(observation, configuration)
    next_node, value = minimax(node, max_depth=3)
    return int(next_node.action)

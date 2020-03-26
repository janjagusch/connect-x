#!/usr/bin/env python


import contextlib as __stickytape_contextlib


@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil

    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)


with __stickytape_temporary_dir() as __stickytape_working_dir:

    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    open(os.path.join(partial_path, "__init__.py"), "w").write("\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "w") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys

    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module("connect_x/__init__.py", "")
    __stickytape_write_module(
        "connect_x/minimax/__init__.py",
        '"""\nThis package provides access to functions and classes for the Minimax algorithm.\n"""\n\nfrom .minimax import minimax\nfrom .tree import ConnectXNode\n',
    )
    __stickytape_write_module(
        "connect_x/minimax/minimax.py",
        '"""\nThis module implementy the Minimax algorithm.\n"""\n\nimport numpy as np\n\n\ndef minimax(node, max_depth=4, maximize=True, current_depth=None):\n    """\n    Executes the Minimax algorithm.\n\n    Args:\n        node (connect_x.minimax.tree.ConnectXNode): The root node.\n        max_depth (int, optional): The maximum recursion depth.\n        maximize (bool, optional): Whether to maximize or minimize.\n        current_depth (int, optional): The current depth.\n\n    Returns:\n        tuple: (Next node to go for, Value of the next node).\n    """\n    current_depth = current_depth or 0\n    value, terminated = node.value\n    if (current_depth == max_depth) or terminated:\n        return value\n    children = node.make_children()\n    values = [\n        minimax(child, max_depth, not maximize, current_depth + 1) for child in children\n    ]\n    if maximize:\n        value = np.max(values)\n        index = np.argmax(values)\n    else:\n        value = np.min(values)\n        index = np.argmin(values)\n    if not current_depth:\n        return children[index], value\n    return value\n',
    )
    __stickytape_write_module(
        "connect_x/minimax/tree.py",
        '"""\nThis module provides tree structures for the Minimax algorithm.\n"""\n\nfrom connect_x.actions import possible_actions, step\nfrom connect_x.evaluate import evaluate\nfrom connect_x.utils import other_mark\n\n\nclass TreeNode:\n    """\n    A generic tree node.\n\n    Args:\n        parent (TreeNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(self, parent=None, children=None):\n        self.parent = parent\n        self.children = children or []\n        if self.parent:\n            self.parent.children.append(self)\n\n\nclass ConnectXNode(TreeNode):\n    """\n    A tree node for Connect X.\n\n    Args:\n        observation (kaggle_environments.utils.Struct): The observation.\n        configuration (kaggle_environments.utils.Struct): The configuration.\n        action (int): The index of the column where you inserted the token.\n        mark (int): The mark of the token.\n        parent (TreeNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(\n        self,\n        observation,\n        configuration,\n        action=None,\n        mark=None,\n        parent=None,\n        children=None,\n    ):\n        super().__init__(parent, children)\n        self.observation = observation\n        self.configuration = configuration\n        self.action = action\n        self.mark = mark\n\n    @property\n    def possible_actions(self):\n        """\n        Returns a list of possible actions you can take from this node.\n\n        Returns:\n            list: List of possible actions.\n        """\n        return possible_actions(self.observation, self.configuration)\n\n    def step(self, action):\n        """\n        Executes an action and returns the new child node.\n\n        Args:\n            action (int): The index of the column where you want to insert the token.\n\n        Returns:\n            ConnectXNode: The new child node.\n        """\n        mark = other_mark(self.mark) or self.observation.mark\n        return self.__class__(\n            observation=step(self.observation, self.configuration, action, mark),\n            configuration=self.configuration,\n            action=action,\n            parent=self,\n            mark=mark,\n        )\n\n    def make_children(self):\n        """\n        Generates all child nodes.\n\n        Returns:\n            list: A list of child nodes.\n        """\n        self.children = [self.step(action) for action in self.possible_actions]\n        return self.children\n\n    @property\n    def value(self):\n        """\n        Calculates the value of the node.\n        """\n        return evaluate(self.observation, self.configuration)\n',
    )
    __stickytape_write_module(
        "connect_x/actions.py",
        '"""\nThis module provides functions to determine possible actions given an observation\nand execute those actions.\n"""\n\nfrom copy import copy\nimport numpy as np\n\nfrom . import utils\n\n\ndef _possible_actions(matrix):\n    """\n    Returns all possible actions you can take from a matrix state.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        np.array: The possible actions.\n    """\n    filter_ = [(array == 0).any() for array in utils.matrix_columns(matrix)]\n    return np.arange(matrix.shape[1])[filter_]\n\n\ndef _step(matrix, action, mark):\n    """\n    Applies an action with a mark to a matrix.\n\n    Args:\n        matrix (np.array): The board matrix.\n        action (int): The column index where the token should be placed.\n        mark (int): The mark of the token.\n\n    Returns:\n        np.array: The new token matrix.\n    """\n    col = matrix[:, action]\n    row = np.argwhere(col == 0).max()\n    new_matrix = matrix.copy()\n    new_matrix[row, action] = mark\n    return new_matrix\n\n\ndef possible_actions(observation, configuration):\n    """\n    Lists all possible actions that can be taken.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n\n    Returns:\n        list: List of possible actions.\n    """\n    board = observation.board\n    n_rows = configuration.rows\n    n_cols = configuration.columns\n\n    matrix = utils.board_to_matrix(board, n_rows, n_cols)\n    return list(_possible_actions(matrix))\n\n\ndef step(observation, configuration, action, mark):\n    """\n    Executes an action and returns the new observation.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n        action (int): The index of the column where you want to insert the token.\n        mark (int): The mark of the token.\n\n    Returns:\n        dict: The new observation.\n    """\n    board = observation.board\n    n_rows = configuration.rows\n    n_cols = configuration.columns\n\n    matrix = utils.board_to_matrix(board, n_rows, n_cols)\n    new_matrix = _step(matrix, action, mark)\n\n    new_board = utils.matrix_to_board(new_matrix)\n\n    new_observation = copy(observation)\n    new_observation.board = new_board\n\n    return new_observation\n',
    )
    __stickytape_write_module(
        "connect_x/utils.py",
        '"""\nThis module contains useful functions for this project.\n"""\n\nimport numpy as np\n\n\ndef other_mark(mark):\n    """\n    Given the mark of a token, returns the other mark.\n\n    Args:\n        mark (int): The mark of the token.\n\n    Returns:\n        int: The other mark or `None`, when mark is `None`.\n    """\n    if not mark:\n        return None\n    assert mark in (1, 2)\n    return 2 if mark == 1 else 1\n\n\ndef board_to_matrix(board, n_rows, n_cols):\n    """\n    Converts a board into a numpy matrix.\n\n    Args:\n        board (list): The board state.\n        n_rows (int): Number of rows on the board.\n        n_cols (int): Number of columns on the board.\n\n    Returns:\n        np.array: The board as a matrix.\n    """\n    return np.array(board).reshape(n_rows, n_cols)\n\n\ndef matrix_to_board(matrix):\n    """\n    Converts a matrix into a board.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        list: The board as a list.\n    """\n    return matrix.reshape(1, -1).tolist()[0]\n\n\ndef rolling_window(array, window_size):\n    """\n    Returns rolling windows over a 1-dimensional array.\n\n    Args:\n        array (np.array): A 1-dimensional arary.\n        window_size (int): The window size.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)\n    strides = array.strides + (array.strides[-1],)\n    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)\n\n\ndef _diagonals(matrix):\n    return [matrix.diagonal(i) for i in range(-matrix.shape[0] + 1, matrix.shape[1])]\n\n\ndef matrix_diagonals(matrix):\n    """\n    Returns all diagonals of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return _diagonals(matrix) + _diagonals(matrix[::-1])\n\n\ndef matrix_rows(matrix):\n    """\n    Returns all rows of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix)\n\n\ndef matrix_columns(matrix):\n    """\n    Returns all columns of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix.T)\n',
    )
    __stickytape_write_module(
        "connect_x/evaluate.py",
        '"""\nThis module contains functions to evaluate environment observations.\n"""\n\nimport numpy as np\n\nfrom . import utils\n\n\ndef _windows(matrix, window_size):\n    """\n    Calculates all windows that are relevant to evaluate to board state from a matrix.\n\n    Args:\n        matrix (np.array): A board matrix.\n        window_size (int): The number of token you need to have \'in a row\'.\n\n    Returns:\n        np.array: The windows of the board.\n    """\n    windows = []\n    # pylint: disable=bad-continuation\n    for array in (\n        utils.matrix_rows(matrix)\n        + utils.matrix_columns(matrix)\n        + utils.matrix_diagonals(matrix)\n    ):\n        # pylint: enable=bad-continuation\n        if len(array) >= window_size:\n            windows.extend(utils.rolling_window(array, window_size))\n    return np.array(windows)\n\n\ndef _eval_windows(windows, mark):\n    """\n    Calculates the evaluation windows, depending on the mark of the token.\n\n    Args:\n        windows (np.array): Array of windows.\n        mark (int): `1` or `2`.\n\n    Returns:\n        np.array: Array of evaluation windows.\n    """\n    mark_opponent = 2 if mark == 1 else 1\n    eval_windows = np.zeros(windows.shape)\n    eval_windows[windows == mark] = 1\n    eval_windows[windows == mark_opponent] = -1\n    return eval_windows\n\n\ndef _evaluate_victory(eval_windows):\n    """\n    Checks whether evaluation windows contain a victory.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether evaluation windows contain victory.\n    """\n    return (eval_windows.mean(axis=1) == 1).any()\n\n\ndef _evaluate_board_full(eval_windows):\n    """\n    Checks whether the board is full.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether the board is full.\n    """\n\n    return not (eval_windows == 0).any()\n\n\ndef _evaluate_heuristic(eval_windows):\n    """\n    Evaluates the board.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: The value of the board.\n    """\n    values = np.exp2(eval_windows.sum(axis=1))\n    not_contains_other = eval_windows.min(axis=1) != -1\n    return (values * not_contains_other).mean()\n\n\ndef _evaluate(eval_windows):\n    """\n    Evaluates the board. Calculates a value for the board and checks whether the game\n    has ended.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    if _evaluate_victory(eval_windows):\n        return float("inf"), True\n    if _evaluate_board_full(eval_windows):\n        return float(0), True\n    return _evaluate_heuristic(eval_windows), False\n\n\ndef evaluate(observation, configuration):\n    """\n    Evaluates an observation.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    mark = observation.mark\n    mark_opponent = 2 if mark == 1 else 1\n\n    matrix = utils.board_to_matrix(\n        observation.board, configuration.rows, configuration.columns\n    )\n    windows = _windows(matrix, configuration.inarow)\n    eval_windows = _eval_windows(windows, mark)\n    eval_windows_opponent = _eval_windows(windows, mark_opponent)\n    value, done = _evaluate(eval_windows)\n    value_opponent, done_opponent = _evaluate(eval_windows_opponent)\n    return value - value_opponent, any([done, done_opponent])\n',
    )
    from connect_x.minimax import minimax, ConnectXNode

    def act(observation, configuration):
        node = ConnectXNode(observation, configuration)
        next_node, value = minimax(node, max_depth=4)
        return int(next_node.action)

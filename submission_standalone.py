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
        '"""\nThis module implementy the Minimax algorithm.\n"""\n\nimport numpy as np\n\n\ndef minimax(node, max_depth=4, maximize=True, current_depth=None):\n    """\n    Executes the Minimax algorithm.\n\n    Args:\n        node (connect_x.minimax.tree.ConnectXNode): The root node.\n        max_depth (int, optional): The maximum recursion depth.\n        maximize (bool, optional): Whether to maximize or minimize.\n        current_depth (int, optional): The current depth.\n\n    Returns:\n        tuple: (Next node to go for, Value of the next node).\n    """\n    current_depth = current_depth or 0\n\n    node.depth = current_depth\n    node.maximize = maximize\n\n    value, terminated = node.value\n    if (current_depth == max_depth) or terminated:\n        return value\n    children = node.make_children()\n    values = [\n        minimax(child, max_depth, not maximize, current_depth + 1) for child in children\n    ]\n    if maximize:\n        value = np.max(values)\n        index = np.argmax(values)\n    else:\n        value = np.min(values)\n        index = np.argmin(values)\n    if not current_depth:\n        return children[index], value\n    return value\n',
    )
    __stickytape_write_module(
        "connect_x/minimax/tree.py",
        '"""\nThis module provides tree structures for the Minimax algorithm.\n"""\n\nfrom connect_x.actions import possible_actions, step\nfrom connect_x.evaluate import evaluate\nfrom connect_x.utils.board import other_mark\n\n\nclass TreeNode:\n    """\n    A generic tree node.\n\n    Args:\n        parent (TreeNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(self, parent=None, children=None):\n        self.parent = parent\n        self.children = children or []\n        if self.parent:\n            self.parent.children.append(self)\n\n\nclass ConnectXNode(TreeNode):\n    """\n    A tree node for Connect X.\n\n    Args:\n        observation (kaggle_environments.utils.Struct): The observation.\n        configuration (kaggle_environments.utils.Struct): The configuration.\n        action (int): The index of the column where you inserted the token.\n        mark (int): The mark of the token.\n        parent (TreeNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(\n        self,\n        observation,\n        configuration,\n        action=None,\n        mark=None,\n        parent=None,\n        children=None,\n    ):\n        super().__init__(parent, children)\n        self.observation = observation\n        self.configuration = configuration\n        self.action = action\n        self.mark = mark\n\n    @property\n    def possible_actions(self):\n        """\n        Returns a list of possible actions you can take from this node.\n\n        Returns:\n            list: List of possible actions.\n        """\n        return possible_actions(self.observation, self.configuration)\n\n    def step(self, action):\n        """\n        Executes an action and returns the new child node.\n\n        Args:\n            action (int): The index of the column where you want to insert the token.\n\n        Returns:\n            ConnectXNode: The new child node.\n        """\n        mark = other_mark(self.mark) or self.observation.mark\n        return self.__class__(\n            observation=step(self.observation, self.configuration, action, mark),\n            configuration=self.configuration,\n            action=action,\n            parent=self,\n            mark=mark,\n        )\n\n    def make_children(self):\n        """\n        Generates all child nodes.\n\n        Returns:\n            list: A list of child nodes.\n        """\n        self.children = [self.step(action) for action in self.possible_actions]\n        return self.children\n\n    @property\n    def value(self):\n        """\n        Calculates the value of the node.\n        """\n        return evaluate(self.observation, self.configuration)\n',
    )
    __stickytape_write_module(
        "connect_x/actions.py",
        '"""\nThis module provides functions to determine possible actions given an observation\nand execute those actions.\n"""\n\nfrom copy import copy\nimport numpy as np\n\nfrom . import utils\n\n\ndef _possible_actions(matrix):\n    """\n    Returns all possible actions you can take from a matrix state.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        np.array: The possible actions.\n    """\n    filter_ = [(array == 0).any() for array in utils.board.matrix_columns(matrix)]\n    return np.arange(matrix.shape[1])[filter_]\n\n\ndef _step(matrix, action, mark):\n    """\n    Applies an action with a mark to a matrix.\n\n    Args:\n        matrix (np.array): The board matrix.\n        action (int): The column index where the token should be placed.\n        mark (int): The mark of the token.\n\n    Returns:\n        np.array: The new token matrix.\n    """\n    col = matrix[:, action]\n    row = np.argwhere(col == 0).max()\n    new_matrix = matrix.copy()\n    new_matrix[row, action] = mark\n    return new_matrix\n\n\ndef possible_actions(observation, configuration):\n    """\n    Lists all possible actions that can be taken.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n\n    Returns:\n        list: List of possible actions.\n    """\n    matrix = utils.board.board_to_matrix(\n        observation.board, configuration.rows, configuration.columns\n    )\n    return list(_possible_actions(matrix))\n\n\ndef step(observation, configuration, action, mark):\n    """\n    Executes an action and returns the new observation.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n        action (int): The index of the column where you want to insert the token.\n        mark (int): The mark of the token.\n\n    Returns:\n        dict: The new observation.\n    """\n\n    matrix = utils.board.board_to_matrix(\n        observation.board, configuration.rows, configuration.columns\n    )\n    new_matrix = _step(matrix, action, mark)\n\n    new_board = utils.board.matrix_to_board(new_matrix)\n\n    new_observation = copy(observation)\n    new_observation.board = new_board\n\n    return new_observation\n',
    )
    __stickytape_write_module("connect_x/utils/__init__.py", "")
    __stickytape_write_module(
        "connect_x/evaluate.py",
        '"""\nThis module contains functions to evaluate environment observations.\n"""\n\nimport numpy as np\n\nfrom . import utils\n\n\ndef _windows(matrix, window_size):\n    """\n    Calculates all windows that are relevant to evaluate to board state from a matrix.\n\n    Args:\n        matrix (np.array): A board matrix.\n        window_size (int): The number of token you need to have \'in a row\'.\n\n    Returns:\n        np.array: The windows of the board.\n    """\n    windows = []\n    # pylint: disable=bad-continuation\n    for array in (\n        utils.board.matrix_rows(matrix)\n        + utils.board.matrix_columns(matrix)\n        + utils.board.matrix_diagonals(matrix)\n    ):\n        # pylint: enable=bad-continuation\n        if len(array) >= window_size:\n            windows.extend(utils.board.rolling_window(array, window_size))\n    return np.array(windows)\n\n\ndef _eval_windows(windows, mark):\n    """\n    Calculates the evaluation windows, depending on the mark of the token.\n\n    Args:\n        windows (np.array): Array of windows.\n        mark (int): `1` or `2`.\n\n    Returns:\n        np.array: Array of evaluation windows.\n    """\n    mark_opponent = 2 if mark == 1 else 1\n    eval_windows = np.zeros(windows.shape)\n    eval_windows[windows == mark] = 1\n    eval_windows[windows == mark_opponent] = -1\n    return eval_windows\n\n\ndef _evaluate_victory(eval_windows):\n    """\n    Checks whether evaluation windows contain a victory.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether evaluation windows contain victory.\n    """\n    return (eval_windows.mean(axis=1) == 1).any()\n\n\ndef _evaluate_board_full(eval_windows):\n    """\n    Checks whether the board is full.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether the board is full.\n    """\n\n    return not (eval_windows == 0).any()\n\n\ndef _evaluate_heuristic(eval_windows):\n    """\n    Evaluates the board.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: The value of the board.\n    """\n    values = np.exp2(eval_windows.sum(axis=1))\n    not_contains_other = eval_windows.min(axis=1) != -1\n    return (values * not_contains_other).mean()\n\n\ndef _evaluate(eval_windows):\n    """\n    Evaluates the board. Calculates a value for the board. and checks whether the game\n    has ended.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    if _evaluate_victory(eval_windows):\n        return float("inf"), True\n    if _evaluate_board_full(eval_windows):\n        return float(0), True\n    return _evaluate_heuristic(eval_windows), False\n\n\ndef evaluate(observation, configuration):\n    """\n    Evaluates an observation.\n\n    Args:\n        observation (dict): The observation.\n        configuration (dict): The configuration.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    mark = observation.mark\n    mark_opponent = utils.board.other_mark(mark)\n\n    matrix = utils.board.board_to_matrix(\n        observation.board, configuration.rows, configuration.columns\n    )\n    windows = _windows(matrix, configuration.inarow)\n    eval_windows = _eval_windows(windows, mark)\n    eval_windows_opponent = _eval_windows(windows, mark_opponent)\n    value, done = _evaluate(eval_windows)\n    value_opponent, done_opponent = _evaluate(eval_windows_opponent)\n    return value - value_opponent, any([done, done_opponent])\n',
    )
    __stickytape_write_module(
        "connect_x/utils/board.py",
        '"""\nThis module contains useful functions for this project.\n"""\n\nimport numpy as np\n\n\nMARK_AGNOSTIC_TOKEN_YOU = 1\nMARK_AGNOSTIC_TOKEN_OTHER = 9\n\n\ndef board_hash(board, mark):\n    """\n    Returns a string representation of the board.\n\n    Args:\n        board (list): The board state.\n        mark (int): Your token mark.\n\n    Returns:\n        str: The board hash.\n    """\n    return "".join(str(val) for val in mark_agnostic_board(board, mark))\n\n\ndef mark_agnostic_board(board, mark):\n    """\n    Makes the board mark agnostic. Replaces your mark with `1` and the other mark with\n    `9`.\n\n    Args:\n        board (list): The board state.\n        mark (int): Your token mark.\n\n    Returns:\n        list: The mark agnostic board.\n    """\n\n    def agnostic(val, mark):\n        if val == 0:\n            return val\n        return MARK_AGNOSTIC_TOKEN_YOU if val == mark else MARK_AGNOSTIC_TOKEN_OTHER\n\n    return [agnostic(val, mark) for val in board]\n\n\ndef game_round(matrix):\n    """\n    Returns in which round the game is.\n\n    Args:\n        matrix (np.array): The board as a matrix.\n\n    Returns:\n        int: The round of the game.\n    """\n    return (matrix != 0).sum()\n\n\ndef middle_column(matrix):\n    """\n    Returns the index of the middle column of the boards.\n\n    Args:\n        matrix (np.array): The board as a matrix.\n\n    Returns:\n        int: The index of the middle column.\n    """\n    _, columns = matrix.shape\n    return int(np.floor(columns / 2))\n\n\ndef other_mark(mark):\n    """\n    Given the mark of a token, returns the other mark.\n\n    Args:\n        mark (int): The mark of the token.\n\n    Returns:\n        int: The other mark or `None`, when mark is `None`.\n    """\n    if not mark:\n        return None\n    assert mark in (1, 2)\n    return 2 if mark == 1 else 1\n\n\ndef board_to_matrix(board, n_rows, n_cols):\n    """\n    Converts a board into a numpy matrix.\n\n    Args:\n        board (list): The board state.\n        n_rows (int): Number of rows on the board.\n        n_cols (int): Number of columns on the board.\n\n    Returns:\n        np.array: The board as a matrix.\n    """\n    return np.array(board).reshape(n_rows, n_cols)\n\n\ndef matrix_to_board(matrix):\n    """\n    Converts a matrix into a board.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        list: The board as a list.\n    """\n    return matrix.reshape(1, -1).tolist()[0]\n\n\ndef rolling_window(array, window_size):\n    """\n    Returns rolling windows over a 1-dimensional array.\n\n    Args:\n        array (np.array): A 1-dimensional arary.\n        window_size (int): The window size.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)\n    strides = array.strides + (array.strides[-1],)\n    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)\n\n\ndef _diagonals(matrix):\n    return [matrix.diagonal(i) for i in range(-matrix.shape[0] + 1, matrix.shape[1])]\n\n\ndef matrix_diagonals(matrix):\n    """\n    Returns all diagonals of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return _diagonals(matrix) + _diagonals(matrix[::-1])\n\n\ndef matrix_rows(matrix):\n    """\n    Returns all rows of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix)\n\n\ndef matrix_columns(matrix):\n    """\n    Returns all columns of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix.T)\n',
    )
    __stickytape_write_module(
        "connect_x/move_catalogue.py",
        '"""\nThis module contains hard-coded rules for Connect X.\n"""\n\nfrom . import utils\n\n\n# pylint: disable=unused-argument\ndef _move_0(matrix, **kwargs):\n    """\n    If you start the game and are first player, always choose the middle column.\n    """\n    return utils.board.middle_column(matrix)\n\n\ndef _move_1(matrix, **kwargs):\n    """\n    If you start the game and are the second player, make sure not to fall for the\n    cheap trick.\n    """\n    middle_column = utils.board.middle_column(matrix)\n    if matrix[-1, middle_column] != 0:\n        return middle_column + 1\n    return middle_column\n\n\n# pylint: enable=unused-argument\n\n\ndef _move_2(matrix, **kwargs):\n    """\n    If you are the first player and it is your second turn, see if you can go for the\n    cheap trick.\n    """\n    middle_column = utils.board.middle_column(matrix)\n    mark = kwargs["mark"]\n\n    if (\n        matrix[-1, middle_column] == mark\n        and matrix[-1, middle_column + 1] == 0\n        and matrix[-1, middle_column - 1] == 0\n        and matrix[-1, middle_column - 2] == 0\n        and matrix[-1, middle_column + 2] == 0\n    ):\n        return middle_column + 1\n    return None\n\n\ndef _move_n(matrix, **kwargs):\n    """\n    Try to complete the cheap trick.\n    """\n    middle_column = utils.board.middle_column(matrix)\n    mark = kwargs["mark"]\n\n    if (\n        matrix[-1, middle_column] == mark\n        and matrix[-1, middle_column + 1] == mark\n        and matrix[-1, middle_column - 1] == 0\n        and matrix[-1, middle_column - 2] == 0\n        and matrix[-1, middle_column + 2] == 0\n    ):\n        return middle_column - 1\n    return None\n\n\n_MOVE_CATALOGUE = {\n    0: _move_0,\n    1: _move_1,\n    2: _move_2,\n    3: _move_2,\n}\n\n\ndef move(observation, configuration):\n    """\n    Makes a move based on the `_MOVE_CATALOGUE`.\n\n    Args:\n        observation (kaggle_environments.utils.board.Struct): The observation.\n        configuration (kaggle_environments.utils.board.Struct): The configuration.\n\n    Return:\n        int: The action.\n    """\n    board = observation.board\n    mark = observation.mark\n    rows = configuration.rows\n    columns = configuration.columns\n    matrix = utils.board.board_to_matrix(board, rows, columns)\n    game_round = utils.board.game_round(matrix)\n\n    move_func = _MOVE_CATALOGUE.get(game_round)\n    if move_func:\n        return move_func(matrix, mark=mark, configuration=configuration)\n    return None\n',
    )
    __stickytape_write_module(
        "connect_x/board_action_map.py",
        '"""\nThis module contains pre-calculated board actions.\n"""\nimport gzip\nimport json\n\nFORECAST_DEPTH = 7\n\n# pylint: disable=line-too-long\n_BOARD_ACTION_MAP_BINARY = b\'\\x1f\\x8b\\x08\\x00nZ\\x7f^\\x02\\xff\\x9d\\x9bAr\\xdc0\\x0c\\x04\\xbf\\xe2\\xf29\\x07qm\\x1d:_K\\xe5\\xefQHJ"\\xd7\\xae\\xf54}\\xc9i\\n"\\xc0\\x80\\xed\\x01\\xfc\\xe7}\\x8b\\x7f\\xde\\x7f\\xbf}\\xfcz{)\\xa0\\xfdSbA\\xff)\\x9c\\x82\\x12\\nZ\\xa8C\\xf0H\\x05U!>\\xa9\\x1d\\xe7\\x10|\\xe6\\x82C\\xe1"\\x1c\\x8a\\\\@\\x89\\xd3:\\x15"Ok/D\\x9e\\xd6^\\x88\\xfc\\x0c\\xbd\\x10yZ{!\\\\\\x04\\x97\\xd6\\xa6\\x10\\x87&\\x17\\x8c\\x85\\x10\\x11Z!\\xc4\\xa1[!DZ[!\\xd4\\\'\\xf5\\xb4>\\xd2\\xb4V\\x858\\x03V\\xd0\\n\\x91\\t\\x86B\\x98\\x08\\xb5\\x10N\\xe0nk+\\x84\\xb9\\xadU!*\\x8d\\x15\\xb4B\\x98\\xabQ\\x0b\\x11\\x9e\\xe1.\\x84\\xf9\\xa4Z\\x08)\\xa8i\\xfd\\xcc\\xd3z(\\xd4m\\x95\\x82V\\x08\\x97V\\x19\\xa1\\x16"=\\xf4U\\x08\\x17\\xe1(\\x84\\xbb\\xad\\x87B\\xa6U\\tZ!dZ]\\x84Z\\x08w\\xf9\\x8eB\\x1c\\x82=\\x12\\x9c\\x85\\xb0i}Y\\xb8\\x19\\xc5\\x02\\xc16\\x0b\\xb0\\x82\\x98\\xc8\\xeeOBFH\\xd1a<42B\\xd6\\x8cy&\\xb2-#2,\\x91a\\x89\\x0cKdX"\\xc3\\x12\\x19\\x17\\x91\\x950\\xad\\x96\\x97\\xb0D\\xe6#X"\\xc3\\x12\\x19\\x96\\xc8\\x88\\x89\\x8cE"\\xc3\\x12\\x19\\x96\\xc8\\xfc\\\'Y"\\xc3\\x12\\x191\\x91\\xb1HdX"\\xe3"\\xb2\\x9cf\\x1c\\x91\\xf93X"#&2\\x16\\x89\\x0cKdX"\\xc3\\x12\\x19\\x96\\xc8\\xeeC\\xef\\x86[\\xe5mu\\xaca\\x89\\x8c\\x8b\\xc8>DZ-\\xfe("\\xe3"\\xb2,\\x02\\x8b\\xdcj<\\xb2\\xdc\\xf2b\\xd1#\\xc3zdX\\x8f\\x0c\\xeb\\x91q\\xa2C1\\xc6`\\x82\\x0e\\xa7\\x9d\\x16\\xd4\\xe1;4N\\x0e=\\xa1\\xb1\\x11\\xc4i\\x9d\\xd08I\\xeb\\x84\\xc6y\\x96\\xb6<\\xad<Y^%\\xb4\\xbc\\xf0\\xa6\\x9a\\xb4\\xbc\\xb0\\x96\\x17\\xd6\\xf2"\\xb6\\xbcXt\\xb0\\xb0\\x0e\\x16\\x0b\\xa6\\x1a\\xda\\x85C\\xdbv\\x08W\\xfa\\xc2\\x9f\\x12gI[^\\xd2\\x90b\\xc5TC\\xbbp\\x91!\\xc5\\x93\\xbf$\\xb2d\\x1d,\\xb4\\xe5\\xe5\\xfc%\\xed\\x91q\\xd2L\\x11Y\\xaa\\xef\\xb4\\xc9\\x92|\\xd8\\xd1$\\xe0\\xec"N8)\\xa1]\\xb4\\xe6\\x91m\\xf9o\\x8af\\xbc\\x86\\x15\\x9c\\x11\\xec<\\x8e\\xde\\xee\\x1fbD\\x88\\x15\\xa8v_d\\x96\\x16\\xc6k\\xd8\\xf1\\x1av\\xbc&\\xb3\\xd4\\x19.\\x12\\x14g\\x0c\\xce\\x0c\\xf7Z\\xf0\\r\\xc3\\x89\\x08\\xd7\\xe5\\xcb\\x05\\xf1\\xe5\\x1b\\x19N\\n\\xc2\\xb4\\x8eH\\x96\\x15\\x8eI \\xc6k\\xe8y\\x1cz\\x80\\x97\\x8d\\x13F\\xc22f?v4\\xc5\\xc2\\xb4\\xcc\\rR\\xe9$\\x90W\\xda\\x0e\\x8e@\\x8f\\xa6\\xc8#\\xb0\\xf4I\\xfda7\\x87vc\\x1d\\xd0\\x8f"2B\\x7f\\xa7\\x1f\\xe9Xgi4\\xe5\\x16m\\xf2\\xf7\\x81mm\\r\\x06\\xac\\xc0\\xae\\xc1\\xe0~\\x7f(\\xf2\\xd0+K*\\xee\\xd0\\x9dN\\xa4\\xc0u\\xbe\\x12?Y#l\\xa8\\x15\\x12y\\xe8\\x06\\x1bR\\x90\\xee:\\x94\\xcd\\x8c\\x08g\\xd8\\x10\\x82\\x08C\\xbf\\xc2\\x86\\x89 \\xee\\xd2\\x00\\x1bR\\x90\\xa6u`\\x07\\xb5\\xaf!\\x047;H\\x81=\\x83]]\\x80\\\\\\xc0Z\\x84x9bx\\xd9\\xe5\\x19\\xac@\\xf9\\xde\\xdb\\xdar\\xc4\\x96\\x8fu\\xdc^%\\x8b\\x8b\\x98\\xa0W=\\xdd\\x0b\\xe4\\xce\\xb0\\xb4\\xf5\\x08z\\xaf2[x\\x1a\\x9f\\xd1\\x10N\\x18\\x05b\\xc5\\x10\\xbbbH\\xbab8\\xbc\\x8a\\xaa\\x91\\x15y\\x86\\xfa*\\xda\\x8d\\xc1tC\\xaa\\x98\\xad\\x96\\xf9U\\x14\\x82\\r+\\x88\\x9e\\xac/\\xaf\\xa2\\x8a\\xd0\\xaf\\x86\\x11\\xc4\\xfb|\\xf7#\\xe7v\\xe1\\x84\\xe0z\\xe4\\xe4\\\'\\xd9=2\\xc8\\x05,E\\xc8X\\x83\\xe9:\\xe5\\xaf\\xe8-\\xd8\\xcd\\x1a\\xbdl\\xc6b\\xa6\\xc8 \\xd8\\xcd\\n\\xba\\x83uc\\xaa1\\x08v\\xb3\\xbe-[\\xa5*\\xdc\\xb6\\xb4\\xc9\\x0cr\\x0b\\x18\\xf9_\\xb4\\xb8,\\xd5\\xf6\\x9dF(\\xc3\\xc6\\xe0\\x9e\\t\\xee\\xc9k*p\\xaetq\\xbf\\x0e\\x0c\\xed\\xfbG\\xc1s\\xfbv\\x11\\xf2J\\xdf\\xddX.\\xc4"1T-m\\xd9?\\xbeA\\x08\\xa6^)\\xc6:\\xe2\\x0fW\\xc6\\xd6g\\x0c)!\\x18:\\x99\\xfa\\xfdA\\x08\\xee\\xc6\\xe4\\xde8!\\xb8\\xfa\\x8c\\xbc|b\\xc9\\xbd \\xefR\\xb9\\xd7/R\\x81\\x98\\x0e\\x8c}\\xc6D0\\x80u\\xf7\\x99\\x9f#<\\xf5\\x99\\xff\\x82\\xbf\\xff\\x00\\xb9\\x84}\\x99w8\\x00\\x00\'\n\n# pylint: enable=line-too-long\nBOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))\n',
    )
    __stickytape_write_module(
        "connect_x/utils/logger.py",
        '"""\nProvides a utils method to setup the logger.\n"""\nimport os\nimport logging\n\n\ndef setup_logger(logger_name=None):\n    """\n    Get the logger.\n\n    Args:\n        logger_name (str): The name of the logger.\n    Returns:\n        logging.Logger: The logger.\n    """\n    logger = logging.getLogger(logger_name)\n    handler = logging.StreamHandler()\n    formatter = logging.Formatter(\n        "%(levelname)s - %(name)s - %(asctime)s - %(message)s",\n        datefmt="%m/%d/%Y %I:%M%S %p",\n    )\n    handler.setFormatter(formatter)\n    logger.addHandler(handler)\n    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))\n    return logger\n',
    )
    """
    This module contains the submissoin for the Kaggle competition.
    """

    import numpy as np

    from connect_x.minimax import minimax, ConnectXNode
    from connect_x.move_catalogue import move
    from connect_x import board_action_map
    from connect_x import utils
    from connect_x.utils.logger import setup_logger

    _LOGGER = setup_logger(__name__)

    FORECAST_DEPTH = 3

    def _game_round(board):
        return (np.array(board) != 0).sum()

    def _rule_based_action(observation, configuration):
        return move(observation, configuration)

    # pylint: disable=unused-argument
    def _precomputed_action(observation, configuration):
        if board_action_map.FORECAST_DEPTH > (
            _game_round(observation.board) + FORECAST_DEPTH
        ):
            board_hash = utils.board.board_hash(observation.board, observation.mark)
            return board_action_map.BOARD_ACTION_MAP[board_hash]
        return None

    # pylint: enable=unused-argument

    def _forecasted_action(observation, configuration):
        node = ConnectXNode(observation, configuration)
        next_node, _ = minimax(node, max_depth=FORECAST_DEPTH)
        return next_node.action

    def act(observation, configuration):
        """
        Decides what action to do next.
    
        Args:
            observation (kaggle_environments.utils.board.Struct): The observation.
            configuration (kaggle_environments.utils.board.Struct): The configuration.
    
        Returns:
            int: The action.
        """
        action = (
            _rule_based_action(observation, configuration)
            or _precomputed_action(observation, configuration)
            or _forecasted_action(observation, configuration)
        )
        return int(action)

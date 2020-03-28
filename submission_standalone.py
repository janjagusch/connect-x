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
        '"""\nThis module provides tree structures for the Minimax algorithm.\n"""\n\nfrom connect_x.actions import possible_actions, step\nfrom connect_x.evaluate import evaluate\nfrom connect_x.utils.board import (\n    other_token,\n    TOKEN_ME,\n)\n\n\nclass TreeNode:\n    """\n    A generic tree node.\n\n    Args:\n        parent (TreeNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(self, parent=None, children=None):\n        self.parent = parent\n        self.children = children or []\n        if self.parent:\n            self.parent.children.append(self)\n\n\nclass ConnectXNode(TreeNode):\n    """\n    A tree node for Connect X.\n\n    Args:\n        matrix (np.array): The board state as matrix.\n        configuration (kaggle_environments.utils.Struct): The configuration.\n        action (int): The index of the column where you inserted the token.\n        next_token (int): The next token (mark independent).\n        parent (ConnectXNode): The parent node.\n        children (list): The child nodes.\n    """\n\n    def __init__(\n        self,\n        matrix,\n        configuration,\n        action=None,\n        next_token=None,\n        parent=None,\n        children=None,\n    ):\n        super().__init__(parent, children)\n        self.matrix = matrix\n        self.configuration = configuration\n        self.action = action\n        self.next_token = next_token\n\n    @property\n    def possible_actions(self):\n        """\n        Returns a list of possible actions you can take from this node.\n\n        Returns:\n            list: List of possible actions.\n        """\n        return possible_actions(self.matrix)\n\n    def step(self, action):\n        """\n        Executes an action and returns the new child node.\n\n        Args:\n            action (int): The index of the column where you want to insert the token.\n\n        Returns:\n            ConnectXNode: The new child node.\n        """\n        next_token = self.next_token or TOKEN_ME\n        return self.__class__(\n            matrix=step(self.matrix, action, next_token),\n            configuration=self.configuration,\n            action=action,\n            next_token=other_token(next_token),\n            parent=self,\n        )\n\n    def make_children(self):\n        """\n        Generates all child nodes.\n\n        Returns:\n            list: A list of child nodes.\n        """\n        self.children = [self.step(action) for action in self.possible_actions]\n        return self.children\n\n    @property\n    def value(self):\n        """\n        Calculates the value of the node.\n        """\n        return evaluate(self.matrix, self.configuration)\n',
    )
    __stickytape_write_module(
        "connect_x/actions.py",
        '"""\nThis module provides functions to determine possible actions given an observation\nand execute those actions.\n"""\n\nimport numpy as np\n\nfrom . import utils\n\n\ndef possible_actions(matrix):\n    """\n    Returns all possible actions you can take from a matrix state.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        list: The possible actions.\n    """\n    filter_ = [(array == 0).any() for array in utils.board.matrix_columns(matrix)]\n    return list(np.arange(matrix.shape[1])[filter_])\n\n\ndef step(matrix, action, token):\n    """\n    Throws a token into a column of the board.\n\n    Args:\n        matrix (np.array): The board matrix.\n        action (int): The column index where the token should be placed.\n        token (int): The token.\n\n    Returns:\n        np.array: The new token matrix.\n    """\n    col = matrix[:, action]\n    row = np.argwhere(col == 0).max()\n    new_matrix = matrix.copy()\n    new_matrix[row, action] = token\n    return new_matrix\n',
    )
    __stickytape_write_module("connect_x/utils/__init__.py", "")
    __stickytape_write_module(
        "connect_x/evaluate.py",
        '"""\nThis module contains functions to evaluate environment observations.\n"""\n\nimport numpy as np\n\nfrom .utils.board import (\n    TOKEN_ME,\n    TOKEN_OTHER,\n    matrix_rows,\n    matrix_columns,\n    matrix_diagonals,\n    rolling_window,\n)\n\n\ndef _windows(matrix, window_size):\n    """\n    Calculates all windows that are relevant to evaluate to board state from a matrix.\n\n    Args:\n        matrix (np.array): A board matrix.\n        window_size (int): The number of token you need to have \'in a row\'.\n\n    Returns:\n        np.array: The windows of the board.\n    """\n    windows = []\n\n    for array in (\n        matrix_rows(matrix) + matrix_columns(matrix) + matrix_diagonals(matrix)\n    ):\n        if len(array) >= window_size:\n            windows.extend(rolling_window(array, window_size))\n    return np.array(windows)\n\n\ndef _eval_windows(windows, mark_me, mark_other):\n    """\n    Calculates the evaluation windows.\n\n    Args:\n        windows (np.array): Array of windows.\n        mark_me (int): The mark of my player.\n        mark_other (int): The mark of the other player.\n\n    Returns:\n        np.array: Array of evaluation windows.\n    """\n    eval_windows = np.zeros(windows.shape)\n    eval_windows[windows == mark_me] = 1\n    eval_windows[windows == mark_other] = -1\n    return eval_windows.astype(int)\n\n\ndef _evaluate_victory(eval_windows):\n    """\n    Checks whether evaluation windows contain a victory.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether evaluation windows contain victory.\n    """\n    return (eval_windows.mean(axis=1) == 1).any()\n\n\ndef _evaluate_board_full(eval_windows):\n    """\n    Checks whether the board is full.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: Whether the board is full.\n    """\n\n    return not (eval_windows == 0).any()\n\n\ndef _evaluate_heuristic(eval_windows):\n    """\n    Evaluates the board.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        bool: The value of the board.\n    """\n    values = np.exp2(eval_windows.sum(axis=1))\n    not_contains_other = eval_windows.min(axis=1) != -1\n    return (values * not_contains_other).mean()\n\n\ndef _evaluate(eval_windows):\n    """\n    Evaluates the board. Calculates a value for the board and checks whether the game\n    has ended.\n\n    Args:\n        eval_windows (np.array): Array of evaluation windows.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    if _evaluate_victory(eval_windows):\n        return float("inf"), True\n    if _evaluate_board_full(eval_windows):\n        return float(0), True\n    return _evaluate_heuristic(eval_windows), False\n\n\ndef evaluate(matrix, configuration):\n    """\n    Evaluates an observation.\n\n    Args:\n        matrix (dict): The matrix of the board state.\n        configuration (dict): The configuration.\n\n    Returns:\n        tuple: (The value of the board, Whether the game has ended).\n    """\n    windows = _windows(matrix, configuration.inarow)\n    eval_windows_me = _eval_windows(windows, TOKEN_ME, TOKEN_OTHER)\n    eval_windows_other = _eval_windows(windows, TOKEN_OTHER, TOKEN_ME)\n    value_me, done_me = _evaluate(eval_windows_me)\n    value_other, done_other = _evaluate(eval_windows_other)\n    return value_me - value_other, any([done_me, done_other])\n',
    )
    __stickytape_write_module(
        "connect_x/utils/board.py",
        '"""\nThis module contains useful functions for this project.\n"""\n\nimport numpy as np\n\n\nTOKEN_ME = 1\nTOKEN_OTHER = 9\n\n\ndef matrix_hash(matrix):\n    """\n    Returns a hash representation of the matrix.\n    Useful for using it as keys in dictionaries.\n\n    Args:\n        matrix (np.array): The board state as matrix.\n\n    Returns:\n        str: The matrix hash.\n    """\n    return matrix.tostring().decode("utf8")\n\n\ndef mark_agnostic_board(board, mark):\n    """\n    Makes the board mark agnostic. Replaces your mark with `1` and the other mark with\n    `9`.\n\n    Args:\n        board (list): The board state.\n        mark (int): Your token mark.\n\n    Returns:\n        list: The mark agnostic board.\n    """\n\n    def agnostic(val, mark):\n        if val == 0:\n            return val\n        return TOKEN_ME if val == mark else TOKEN_OTHER\n\n    return [agnostic(val, mark) for val in board]\n\n\ndef game_round(matrix):\n    """\n    Returns in which round the game is.\n\n    Args:\n        matrix (np.array): The board as a matrix.\n\n    Returns:\n        int: The round of the game.\n    """\n    return (matrix != 0).sum()\n\n\ndef middle_column(matrix):\n    """\n    Returns the index of the middle column of the boards.\n\n    Args:\n        matrix (np.array): The board as a matrix.\n\n    Returns:\n        int: The index of the middle column.\n    """\n    _, columns = matrix.shape\n    return int(np.floor(columns / 2))\n\n\ndef other_token(token):\n    """\n    Given a token, returns the other token.\n\n    Args:\n        token (int): The token.\n\n    Returns:\n        int: The other token or `None`.\n    """\n    if not token:\n        return None\n    assert token in (TOKEN_ME, TOKEN_OTHER)\n    return TOKEN_OTHER if token == TOKEN_ME else TOKEN_ME\n\n\ndef board_to_matrix(board, n_rows, n_cols):\n    """\n    Converts a board into a numpy matrix.\n\n    Args:\n        board (list): The board state.\n        n_rows (int): Number of rows on the board.\n        n_cols (int): Number of columns on the board.\n\n    Returns:\n        np.array: The board as a matrix.\n    """\n    return np.array(board).reshape(n_rows, n_cols)\n\n\ndef matrix_to_board(matrix):\n    """\n    Converts a matrix into a board.\n\n    Args:\n        matrix (np.array): The board matrix.\n\n    Returns:\n        list: The board as a list.\n    """\n    return matrix.reshape(1, -1).tolist()[0]\n\n\ndef rolling_window(array, window_size):\n    """\n    Returns rolling windows over a 1-dimensional array.\n\n    Args:\n        array (np.array): A 1-dimensional arary.\n        window_size (int): The window size.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    shape = array.shape[:-1] + (array.shape[-1] - window_size + 1, window_size)\n    strides = array.strides + (array.strides[-1],)\n    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)\n\n\ndef _diagonals(matrix):\n    return [matrix.diagonal(i) for i in range(-matrix.shape[0] + 1, matrix.shape[1])]\n\n\ndef matrix_diagonals(matrix):\n    """\n    Returns all diagonals of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return _diagonals(matrix) + _diagonals(matrix[::-1])\n\n\ndef matrix_rows(matrix):\n    """\n    Returns all rows of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix)\n\n\ndef matrix_columns(matrix):\n    """\n    Returns all columns of a 2-dimensional matrix.\n\n    Args:\n        matrix (np.array): A 2-dimensional matrix.\n\n    Returns:\n        list: List of np.array objects.\n    """\n    return list(matrix.T)\n',
    )
    __stickytape_write_module(
        "connect_x/move_catalogue.py",
        '"""\nThis module contains hard-coded rules for Connect X.\n"""\n\nfrom .utils.board import (\n    middle_column,\n    game_round,\n    board_to_matrix,\n    TOKEN_ME,\n)\n\n\n# pylint: disable=unused-argument\ndef _move_0(matrix, **kwargs):\n    """\n    If you start the game and are first player, always choose the middle column.\n    """\n    return middle_column(matrix)\n\n\ndef _move_1(matrix, **kwargs):\n    """\n    If you start the game and are the second player, make sure not to fall for the\n    cheap trick.\n    """\n    middle_column_ = middle_column(matrix)\n    if matrix[-1, middle_column_] != 0:\n        return middle_column_ + 1\n    return middle_column_\n\n\ndef _move_2(matrix, **kwargs):\n    """\n    If you are the first player and it is your second turn, see if you can go for the\n    cheap trick.\n    """\n    middle_column_ = middle_column(matrix)\n\n    if (\n        matrix[-1, middle_column_] == TOKEN_ME\n        and matrix[-1, middle_column_ + 1] == 0\n        and matrix[-1, middle_column_ - 1] == 0\n        and matrix[-1, middle_column_ - 2] == 0\n        and matrix[-1, middle_column_ + 2] == 0\n    ):\n        return middle_column_ + 1\n    return None\n\n\n# pylint: enable=unused-argument\n\n\n_MOVE_CATALOGUE = {\n    0: _move_0,\n    1: _move_1,\n    2: _move_2,\n    3: _move_2,\n}\n\n\ndef move(matrix, configuration):\n    """\n    Makes a move based on the `_MOVE_CATALOGUE`.\n\n    Args:\n        matrix (np.array): The board state as matrix.\n        configuration (kaggle_environments.utils.Struct): The configuration.\n\n    Return:\n        int: The action.\n    """\n\n    move_func = _MOVE_CATALOGUE.get(game_round(matrix))\n    if move_func:\n        return move_func(matrix, configuration=configuration)\n    return None\n',
    )
    __stickytape_write_module(
        "connect_x/board_action_map.py",
        "\"\"\"\nThis module contains pre-calculated board actions.\n\"\"\"\nimport gzip\nimport json\n\nFORECAST_DEPTH = 7\n\n# pylint: disable=line-too-long\n_BOARD_ACTION_MAP_BINARY = b\"\\x1f\\x8b\\x08\\x00\\xa5\\xb1\\x7f^\\x02\\xff\\xed\\xdd=r\\x93A\\x16\\x85\\xe1\\xadP\\x8e\\t\\xdc\\xc2NX\\x0b\\xcbh\\\"\\x8a\\xbd\\xa3\\x15\\xb8J\\xd8\\x92\\xfa\\x9e\\xf3L\\xf0\\xc4\\xcc7\\xd4\\xbc}\\xfbG\\xfcy\\xf9\\xf5\\xfb\\xf5\\xfa\\x1f\\x92$\\x99\\xe4\\xcb\\xcfo?\\xbe\\x7f\\x93y\\x92\\xfc_\\xb7o\\xf0\\xa5.\\xdfA\\xe2I\\x922o\\x89\\xf5q\\xe2\\x97\\xc4\\x93$[&\\xf9]\\x95\\xf8\\x8b\\xc4\\x93$m\\xd6\\xc7-\\tl\\xd4\\x93$%>rIpM\\xfc\\x9b\\xc4\\x93$%>nI`\\x8a'IJ|\\xe4\\x92@\\xe2I\\x92\\x93\\xa7Z\\xcb\\x01\\x89'Ig\\xb5&\\xfe\\xba\\xc4{4G\\x92\\xac\\xc9\\xbcGs$I\\x9a\\xe4=\\x9a#I\\xd2f\\xbdGs$I:\\x8f\\xf7h\\x8e$IW\\xee<\\x9a#Izq`9\\xe0\\xba\\x1dI\\xd2;z\\x89'I\\xba\\xbeeS_\\xe2I\\x92\\xf4\\x8e\\xde\\xa39\\x92$\\xbd\\xa3\\xbfc\\xe2=\\x9a#IJ|\\xdeA\\x8c\\x8dz\\x92\\xa4\\xc4\\xa7>\\x9a\\x93x\\x92\\xa4\\xeb\\x85y\\xcb\\x01g\\xf1$IK\\x02\\x89'I\\xb2lI \\xf1$I\\x1b\\xdd\\xce\\xed%\\x9e$I\\xef\\xe8%\\x9e$I\\xef\\xe8%\\x9e$I\\xef\\xe8%\\x9e$\\xe9.A\\xf2r\\xc0\\xaf\\xdb\\x91$-\\t$\\x9e$IK\\x82A\\x89\\xf7\\xebv$I~\\xd1\\x92\\xe0\\xac\\xc4;\\x8b'I\\t\\xa3\\x8dz\\x92$9\\xe2\\x1d\\xbd\\xc4\\x93$\\x19\\xf9\\x8e^\\xe2I\\x92\\x0e\\x0e\\\"\\x97\\x03\\xce\\xe2I\\x92\\x96\\x04\\x12O\\x92\\xa4%\\xc1\\xa0\\xc4{4G\\x92\\xe4\\x93\\x96\\x04\\xa6x\\x92$\\xbd\\xa3w\\xa3\\x9e$\\xe9\\xec\\xdbY\\xbc)\\x9e$\\xc9\\xccw\\xf4\\x12O\\x92\\xb4K\\x10\\xb9\\x1c\\x90x\\x92\\xa4%\\x81\\xc4\\x93$iI0(\\xf1\\x1e\\xcd\\x91$9dI`\\x8a'I\\xb2\\xfd\\x1d\\xbdGs$IF\\xbe\\xa3\\xbf&\\xfe]\\xe2I\\x92\\xce\\xbe]\\xb7#I\\xd2\\x92`\\x82\\x12O\\xfa\\xbfP>l\\xaa\\xa2\\xbf\\x9b\\xf2N\\x92\\x12/\\xf1\\x96\\x03\\xf2N\\x92\\xf2n\\x82\\xb7\\x1c\\xf0\\x16\\x9e$\\xe5\\xdd\\x06}\\xe8r\\xc0\\xf4N\\x92\\xf2.\\xefq\\x7fvo\\xe0IR\\xde%2\\xee\\xbf\\xab\\xe9\\x9d$\\xe5]\\xde\\xe3\\xbe\\x8d\\xbc\\x93\\xa4\\x1cM\\xfd\\x96\\x96U\\x1f\\xe5\\xfdU\\xdeI\\xd2r@\\xde\\xd3\\xf2\\xbe\\xe4\\x9d$\\xd9\\x90x\\x0f\\xe3H\\x92\\xac\\x9f\\xe0=\\x8c#I\\xb2~\\x83\\xde\\xc38\\x92$\\xeb\\xcf\\xdf=\\x8c#I\\xca{\\xfd\\xf5\\xba\\xdb\\x1e\\xc6\\xb9ZG\\x92n\\xc2W|\\xcb\\xa2\\xa5\\x80\\xabu$\\xc9\\xa9\\x897\\xe9\\xcb;I\\x9ahM\\xef\\xf2N\\x92d{\\xe2=\\x8c#I\\xb2~\\x82\\xf70\\x8e$\\xc9\\xfa\\r\\xfams\\x9e$)\\xef\\xed\\xe7\\xef[\\xdeI\\xd2\\xd57\\xdf\\xb2x)\\xe0\\xec\\x9d$\\xd9\\xf20N\\xdeI\\x92\\x8c\\x9b\\xe0\\xb7\\xbc\\x93$\\xe5\\xd1\\xe6\\xbc\\xbc\\x93$Y\\x9e\\xf8-\\xef$I\\xb6O\\xf0\\xfe\\xc58\\x92\\xa4\\xbc\\xd7o\\xd0oy'Ig\\xdd\\xbee\\xf1R\\xc0\\xaf\\xd6\\x91$=\\x8c\\x93w\\x92$K\\x1e\\xc6\\xc9;I\\x92q\\x1b\\xf4{t\\xde\\x9d\\xbd\\x93\\xa4\\x9c\\xfa\\xb3\\x98\\xdeI\\x92\\x94\\xf8\\xa3\\xdf\\xc9\\xcb;IR\\xdegL\\xf0[\\xdeIR\\x8e|\\xcb\\xe2\\xa5\\x80\\xb3w\\x92\\xa4\\x87q\\xf2N\\x92\\xa4\\x87q\\x03\\xf2~\\x91w\\x92\\xa4\\r\\xfa\\xa8w\\xf2\\xa6w\\x92\\xa4\\xbc?\\xe7\\xcf~\\xdf\\xbc\\xbbZG\\x92\\xb4y\\x1e\\xb79\\xff.\\xef$I\\xd3{\\xd4;y\\x9b\\xf3$)G\\xbee\\xdcR@\\xdeI\\x92\\x1e\\xc6\\xc9;I\\x92\\xce\\xf6\\xe5\\x9d$I\\x0f\\xe3\\xe4\\x9d$)\\xef\\xfe\\xec\\x9f\\xce\\xbb\\x87q$Iy\\xcfz'oz'I\\xba*\\xe7\\xec\\x9d$I\\xd3\\xfe\\x04%\\x9e$%f\\xe8\\x94\\xea\\xdb\\x7f\\x98\\xf7%\\xef$i9 \\xef\\xf2N\\x92\\xe4\\xc0\\xc4\\xef\\xaa\\xbc\\xfbGaI\\x92&\\xf8\\xac\\xe5\\x80\\xb3w\\x92\\xa4\\xbc\\xc7-\\x07\\xbc}'I\\xca{\\xdcr\\xc0\\xd9;IR\\xde\\xe3\\x96\\x03\\xf2N\\x92g\\xe7\\xcb\\xcb\\x02iw\\xf6N\\xd2\\xf5\\\"i\\xf7w\\x93\\xf2N\\x92\\x12o\\xda\\x97w\\x92\\xa4\\xbc\\xd7L\\xf0\\x1e\\xc6\\x91$\\xe5\\xbd~\\x83\\xde\\xc38\\x92\\x94w\\xdf\\xa1\\xfe\\xfc\\xdd\\xc38\\x92\\x94wz\\x18'\\xef$)G\\xbee\\xf1\\xb2J\\xdeI\\xd2r@\\xde\\x13\\xf3\\xee\\x9f\\x94!IJ\\xbc\\x9b\\xf3$I\\x9a\\xe0\\xdd\\x9c'I\\xd2\\x06\\xfd\\x83o\\xce\\xdb\\x9c'I\\xca{\\xdc\\xcdyy'IW\\xe5|\\xcb\\xac\\xa5\\x80\\xb3w\\x92\\xa4\\xdf\\x9c\\x97w\\x92\\xa4\\xdd\\x01\\xd3\\xbb\\xbc\\x93$\\xd9\\x91\\xf8-\\xef$I\\xb6O\\xf0[\\xdeI\\x92l\\xdf\\xa0\\xdf\\xf2N\\x92\\xce\\xc6}\\xcbG\\xbb\\x8e\\xda\\x9c\\xf7\\x9b\\xf3$I7\\xe7\\xe3\\xf2\\xeeW\\xebH\\x92&\\xf8\\xac\\x87t6\\xe7IR\\x1e\\xfd\\xd9M\\xef$IJ\\xfc\\xe9\\x0f\\xe9\\xe4\\x9d$)\\xef3&\\xf8-\\xef$)G\\xbe\\xe5\\xe9\\xae\\xbbN\\xefn\\xce\\x93$\\xdd\\x9c\\x8f\\xcb\\xbb\\xabu$I\\x13|\\xd6C:y'I\\xca\\xfbs\\xfe,\\xf7\\xcd\\xbb\\xb3w\\x92dEN_m\\xce\\x93$i\\xb9qV\\xe2o\\xbb9\\xefj\\x1dI\\xca\\x91o\\x99\\xb5\\x14\\x90w\\x92\\xa4\\x9b\\xf3\\x89y\\xb79O\\x924\\xc1g=\\xa4\\x93w\\x92\\xa4\\xbc\\xcf\\xf8\\xb3\\xdc\\x96w7\\xe7I\\x92\\xf2\\x9e\\xf5\\x90\\xce\\xd9;I\\xd2\\xe6\\xb9\\xb3w\\x92$-7&xM\\xfcE\\xe2IR2\\n~\\xeeE\\xdeI\\x92\\x96\\x0f\\xf2.\\xef$I\\xc6%~\\x8f\\xce\\xbb\\xf3w\\x92\\xa4\\xbcg\\xed\\x0e\\x98\\xdeI\\x92\\xf2\\x1ew\\xf0\\\"\\xef$Iy\\x8f\\xbb\\x87!\\xef$\\xe9\\xaa\\\\\\xcbR\\xc0\\xd5:\\x92\\xa4\\xe5\\x80\\xbc\\xcb;I\\x92i\\x89ws\\x9e$\\xc9\\xfa\\t\\xfe\\xac\\x9b\\xf3\\xfeY\\x19\\x92\\xa4\\xbc\\xbb9O\\x92\\xa4\\xbc\\xbb9O\\x92<\\xf6\\xa1X\\xd3UBW\\xebH\\x92\\xf2\\x1e\\xb7\\x1c\\x90w\\x924!I\\xbb\\xbf\\x9b\\xf2N\\x92\\x94x\\xd3\\xbe\\xbc\\x93$\\xe5\\xdd\\xd9\\xbe\\xbc\\x93\\xa4\\xbc3k\\x83\\xde\\xc38\\x92\\x94w\\xdf\\xa1\\xfe\\xfc\\xdd\\xc38\\x92\\x94#~\\xfa[\\x0e^V\\xc9;IZ\\x0e\\xf86\\x89y\\xf7\\x9b\\xf3$I\\x89\\xcf\\xfaE<y'I\\xca\\xfbs&\\xf8-\\xef$I\\xb6o\\xd0oy'Ig\\xe9|\\xf4\\xb7\\\\Gm\\xce\\xbbZG\\x92\\x94\\xf8\\xacI\\xdf\\xf4N\\x92r\\xe7\\xbfk\\xe2\\xf4\\xfe&\\xef$I\\x89\\x7f|\\xe2\\xfdj\\x1dI\\x92\\xf5\\x13\\xbc_\\xad#I9\\xe2\\xf1\\xdfr\\x99\\xdeI\\x92<\\xfdj\\xddYyw\\xb5\\x8e$)\\xefYo\\xeaM\\xef$\\xc9\\x96\\xc3\\x02\\x9b\\xf3$IZ\\x9e\\x1c\\x96xW\\xebHR\\x8e\\x18\\xf7-\\x97\\xbc\\x93$y\\xfa\\xd5\\xba\\xfb\\xe6\\xdd\\xd5:\\x92\\xa4\\xbcg\\xbd\\xa97\\xbd\\x93$\\xe5}\\xc6\\x9fE\\xdeI\\x92r\\xea\\xec]\\xdeI\\x92\\x0c[\\x128\\x7f'I\\xb7\\xdb\\x87N\\xa9\\xfe\\xb7\\x95w\\x92\\xb4\\xdc\\x90wy'IR\\xe2\\xe7>\\xa6\\x93w\\x92\\xa4\\xbc\\xc7=\\xa6\\x93w\\x92\\xa4\\xbc\\xc7\\xdd\\xfe\\x97w\\x92t\\x9ek)\\xe0\\xec\\x9d$i\\xf9 \\xef\\xf2N\\x92dG\\xe2]\\xad#I\\xb2~\\x82w\\xb5\\x8e$\\xc9\\xfa\\rzW\\xebH\\xd2Y:?\\xbd\\x14\\x98}\\xf6\\xeew\\xe7I\\xd2r@\\xde]\\xad#IR\\xe2]\\xad#I\\xd2\\x04\\xefj\\x1dI\\x926\\xe8]\\xad#I\\x1e\\xfa\\xa3\\xae%7\\xd5\\x9d\\xbd\\x93$\\xe5\\xbd\\xfe]\\xba\\xbc\\x93$\\xe5\\xbd\\xfeW\\xe8\\xe4\\x9d\\xa4\\xe9K\\xda}\\x07\\xd3\\xbb\\xbc\\x93\\xa4\\xc4\\xd3\\xc38\\x92\\xa4\\xbc\\xcb\\xbb\\x87q$Iy\\x97w\\x0f\\xe3H\\x92\\xee2L\\xfc\\x96\\xcb\\xe6<I\\xd2r@\\xde\\xe5\\x9d$\\xc9\\xb0\\xc4;{'I\\xb2~\\x82w\\xf6N\\x92r\\xc4\\xe3\\xbf\\xe52\\xbd\\x93$\\x99\\xf6\\xee]\\xdeI\\x92v\\x07L\\xef\\xf2N\\x92dZ\\xe2\\x9d\\xbd\\x93\\xa4\\x1c1\\xee[.y'I2\\xed7\\xe7\\xe5\\x9d$\\xc9\\xb8\\xdf\\x9c\\x97w\\x92\\xa4\\rn\\x9b\\xf3\\xf2N\\x92d\\xd8\\x92\\xe0\\x9a\\xf87\\x89'I\\xd7\\xd9\\x06N\\xa9\\xfe.\\xc8;I2n\\x13\\xda\\xb7\\x94w\\x92\\xa4\\t\\xbe\\xe8\\xb6\\xbd\\xf3w\\x92\\xa4\\xbc\\xc7\\xdd\\xb6\\x97w\\x92tVo)`s\\x9e$i\\xb9!\\xef\\xf2N\\x92\\xa4\\xc4;{'I\\xd2\\x04\\xff\\xe9\\xb3w\\xd3;I\\xda<\\xb7\\x14\\x88\\xdb\\x9c\\xbf\\xc8;IZ>\\xc8\\xbb\\xb3w\\x92$%\\xfe\\xf0\\xb3w\\xd3;IR\\xde\\xe3\\xce\\xde\\xe5\\x9d$m\\xb6\\xf3\\x0b\\x96\\x02gm\\xce\\xbb9O\\x92\\x96\\x03\\x8c\\xcb\\xbb\\xb3w\\x92$\\x9f\\x90x\\xef\\xdeI\\x92\\xac\\x9f\\xe0\\xfd\\xe6<I\\xd2\\xcf\\xc24\\xff\\xd9m\\xce\\x93$\\xfd\\x0bm\\xf2N\\x92\\x94w\\xcb\\x01y'I\\xca{\\xc7r\\xc0\\xc38\\x92\\xf4\\x90K\\xda\\xfd\\xdd4\\xbd\\x93$%\\xde\\xb4/\\xef$Iy\\xef=\\xdb\\x97w\\x92\\x94#_\\xe29\\xdfr\\x99\\xdeI\\x92\\x96\\x03\\xf2.\\xef$I\\x86%\\xde\\xe6<I\\xd24\\x1e\\xf7-\\x97\\xbc\\x93$\\x99vs^\\xdeI\\x92v\\x07L\\xef\\xf2N\\x92d\\xd8\\x92\\xc0/\\xd7\\x91\\xa4\\t\\xf5$\\xfd\\x06\\xbe\\xbc\\x93$\\x9b7\\xa1}\\xcb\\x0f\\xf3\\xfe.\\xef$I\\x13|\\xd4u<\\xd3;I\\xda\\xcc\\xb7\\x14H\\x9c\\xde\\x97\\xbc\\x93$\\xa7.O\\xe4\\xdd\\xe6<I\\xd2\\x04os\\x9e$iZ\\xb6\\x14\\x98:\\xbd_\\xe4\\x9d$\\xd9\\xb2\\xdcX6\\xe7I\\x924\\xc1O\\xdd\\x9c7\\xbd\\x93\\xa4\\xe9\\x9aOX\\n\\xf8Y\\x1b\\x92\\xa4\\xe5\\x83\\xbc\\xcb;I\\x92i\\x89ws\\x9e$i\\x1a?~)p\\xdf\\xe9\\xdd\\xbf\\x1aG\\x92\\xb4\\x1c\\x88\\xcb\\xbb\\x9b\\xf3$I\\x0eH\\xfc6\\xbd\\x93$]7+\\xde\\t1\\xbd\\x93$\\xfd4\\x8c\\xbc\\x93$\\xe5]R\\xfd(-IR\\xde-\\x07\\xbc{'I\\xca{\\xc5r@\\xdeI\\xd2U-i\\xf7wS\\xdeI\\x92\\x12_<\\xed\\xcb;I\\xd2n\\xc8\\x8co\\xb9n\\xca\\xbb\\x9b\\xf3$I\\xcb\\x81\\xb8\\xbc\\x9b\\xdeI\\x92\\xcc[\\x12H<I\\x9aP\\x9d\\xed\\xcb;I\\x92glB\\xfb\\x96\\xf2N\\x92\\xa6}\\xd3~W\\xde\\x97\\xbc\\x93$\\xa7.g\\xe4\\xdd\\xf4N\\x92\\xf2h\\xda\\xaf\\xc9\\xbb\\xdf\\x9d'I\\xd6,O\\xe4\\x9d$)\\xa7\\xa6}\\x9b\\xf3$I\\xa6-7\\xe4\\x9d$)\\xbf\\xa6\\xfdc\\xf2\\xfe&\\xef$I\\xcb\\x07y'I\\xca5\\x1f\\xbe\\x1c\\xb8-\\xef\\xfeY\\x19\\x92\\xa4\\xe5@\\\\\\xde\\x9d\\xbd\\x93$'&\\xccRI\\xdeI\\x92\\xf5\\xe7\\xd1\\xf2N\\x92\\xe4\\x8c\\xbc7-\\x07\\xfc\\xac\\rIR\\xde\\xe3\\x96\\x03\\xa6w\\x92\\xa4\\xbc\\xc7-\\x07<\\x8c#I\\xca{\\xdcr\\xc0\\xf4N\\x92\\xaejI{\\xfb\\xd5\\xba\\xbf\\xff\\x00-e\\xd0\\xfd\\xe1\\x12\\t\\x00\"\n\n# pylint: enable=line-too-long\nBOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))\n",
    )
    __stickytape_write_module(
        "connect_x/utils/logger.py",
        '"""\nProvides a utils method to setup the logger.\n"""\nimport os\nimport logging\n\n\ndef setup_logger(logger_name=None):\n    """\n    Get the logger.\n\n    Args:\n        logger_name (str): The name of the logger.\n    Returns:\n        logging.Logger: The logger.\n    """\n    logger = logging.getLogger(logger_name)\n    handler = logging.StreamHandler()\n    formatter = logging.Formatter(\n        "%(levelname)s - %(name)s - %(asctime)s - %(message)s",\n        datefmt="%m/%d/%Y %I:%M%S %p",\n    )\n    handler.setFormatter(formatter)\n    logger.addHandler(handler)\n    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))\n    return logger\n',
    )
    """
    This module contains the submissoin for the Kaggle competition.
    """

    from connect_x.minimax import minimax, ConnectXNode
    from connect_x.move_catalogue import move
    from connect_x.board_action_map import (
        FORECAST_DEPTH as PRECOMPUTED_DEPTH,
        BOARD_ACTION_MAP,
    )
    from connect_x.utils.board import (
        game_round,
        mark_agnostic_board,
        matrix_hash,
        board_to_matrix,
    )
    from connect_x.utils.logger import setup_logger

    _LOGGER = setup_logger(__name__)

    FORECAST_DEPTH = 3

    def _rule_based_action(matrix, configuration):
        return move(matrix, configuration)

    # pylint: disable=unused-argument
    def _precomputed_action(matrix, configuration):
        if PRECOMPUTED_DEPTH > (game_round(matrix) + FORECAST_DEPTH):
            return BOARD_ACTION_MAP[matrix_hash(matrix)]
        return None

    # pylint: enable=unused-argument

    def _forecasted_action(matrix, configuration):
        node = ConnectXNode(matrix, configuration)
        next_node, _ = minimax(node, max_depth=FORECAST_DEPTH)
        return next_node.action

    def act(observation, configuration):
        """
        Decides what action to do next.
    
        Args:
            observation (kaggle_environments.utils.Struct): The observation.
            configuration (kaggle_environments.utils.Struct): The configuration.
    
        Returns:
            int: The action.
        """
        board = mark_agnostic_board(observation.board, observation.mark)
        matrix = board_to_matrix(board, configuration.rows, configuration.columns)
        action = (
            _rule_based_action(matrix, configuration)
            or _precomputed_action(matrix, configuration)
            or _forecasted_action(matrix, configuration)
        )
        return int(action)

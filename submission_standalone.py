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
        '"""\nThis module contains pre-calculated board actions.\n"""\nimport gzip\nimport json\n\nFORECAST_DEPTH = 5\n\n# pylint: disable=line-too-long\n_BOARD_ACTION_MAP_BINARY = b"\\x1f\\x8b\\x08\\x00\\xc7\\xad\\x7f^\\x02\\xff\\xed\\xd7\\xb1\\r\\x800\\x10\\x04\\xc1V,\\xc7\\x04\\x08\\x88\\\\\\x0be@\\x84\\xe8\\x9d\\xa7\\x0bt\\x8c\\x83)\\xc0\\xc9\\xde_}?\\xe7z$I2\\xc9>\\xda:5\\x99\'I~\\xc3\\xc3\\x1fH<I\\x92&\\x81\\xc4\\x93$\\xf9\\xb7IP\\x89_$\\x9e$\\xc9\\xb8I\\xe0\\x8a\'I2r\\x12T\\xe27\\x89\'I2n\\x12\\xb8\\xe2I\\x92\\x8c\\x9c\\x04o\\xe2\\xef\\x07\\x83\\xc4(y\\x1c?\\x00\\x00"\n\n# pylint: enable=line-too-long\nBOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))\n',
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

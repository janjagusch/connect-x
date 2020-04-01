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
        "connect_x/game/__init__.py",
        '"""\nImports to faciliate access from root.\n"""\n\nfrom .connect_x import ConnectXGame, ConnectXState\n',
    )
    __stickytape_write_module(
        "connect_x/game/connect_x.py",
        '"""\nThis module defines the Connect-X game.\n"""\n\nfrom copy import deepcopy\n\nimport numpy as np\nfrom kaggle_environments.utils import Struct\n\nfrom connect_x.game.game import Game, GameState\nfrom connect_x.utils.converter import (\n    board_to_bitmaps,\n    bitmaps_to_matrix,\n    bitmaps_to_board,\n)\n\n\nclass ConnectXState(GameState):\n    """\n    This class represent a game state for Connect-X.\n\n    Args:\n        bitmaps (list): The board, represented as a list of two integers.\n        action_log (list): The log of previously executed actions.\n        height (list): The height of each column in the board,\n            in bitmap representation.\n        counter (int): The number of turn already played.\n        mark (int): The mark of the player.\n\n    """\n\n    def __init__(self, bitmaps, mark, action_log=None, height=None, counter=None):\n        self.bitmaps = bitmaps\n        self.mark = mark\n        self.action_log = action_log or []\n        self.height = height or self._update_height(bitmaps)\n        self.counter = counter or self._update_counter(bitmaps)\n\n    @classmethod\n    def from_observation(cls, observation, rows, columns):\n        """\n        Creates a ConnectXState from an observation.\n\n        Args:\n            observation (kaggle_environments.utils.Struct): The observation.\n            rows (int): The number of rows.\n            columns (int): The number of columns.\n\n        Returns:\n            ConnectXState: The state.\n        """\n        assert rows == 6, "The state only supports rows=6 for now."\n        assert columns == 7, "The game only supports columns=7 for now."\n        return cls(\n            bitmaps=board_to_bitmaps(observation.board, rows, columns),\n            mark=observation.mark,\n        )\n\n    def to_observation(self):\n        """\n        Creates an observation from the state.\n\n        Returns:\n            kaggle_environments.utils.Struct: The observation.\n        """\n        return Struct(board=bitmaps_to_board(self.bitmaps), mark=self.mark,)\n\n    @staticmethod\n    def _update_height(bitmaps):\n        additional_height = (bitmaps_to_matrix(bitmaps) != 0).sum(axis=0)\n        base_height = np.array([i * 7 for i in range(7)])\n        return list(base_height + additional_height)\n\n    @staticmethod\n    def _update_counter(bitmaps):\n        return bin(bitmaps[0])[2:].count("1") + bin(bitmaps[1])[2:].count("1")\n\n    @property\n    def state_hash(self):\n        """\n        Creates a unique hash for the state.\n\n        Returns:\n            int: The state hash.\n        """\n        return 2 * self.bitmaps[0] + self.bitmaps[1]\n\n    def __repr__(self):\n        attr_str = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())\n        return f"{self.__class__.__name__}({attr_str})"\n\n    def __eq__(self, other):\n        return self.bitmaps == other.bitmaps\n\n\nclass ConnectXGame(Game):\n    """\n    This class represents the Connect-X game.\n\n    Args:\n        rows (int): The number of rows.\n        columns (int): The number of columns.\n        x (int): The number of tokens connected to win.\n        timeout (int): The timeout for the turn.\n        steps (int): The maximum number of steps.\n    """\n\n    _STATE_CLS = ConnectXState\n\n    def __init__(self, rows=6, columns=7, x=4, timeout=None, steps=1000):\n        assert rows == 6, "The game only supports rows=6 for now."\n        assert columns == 7, "The game only supports columns=7 for now."\n        assert x == 4, "The game only supports x=4 for now."\n        self.rows = rows\n        self.columns = columns\n        self.x = 4\n        self._timeout = timeout\n        self._steps = steps\n        self._actions = range(columns)\n        self._top = int("_".join("1000000" for _ in range(columns)), 2)\n\n    @classmethod\n    def from_configuration(cls, configuration):\n        """\n        Creates a game from a configuration.\n\n        Args:\n            configuration (kaggle_environments.utils.Struct): The configuration.\n\n        Returns:\n            ConnectXGame: The game.\n        """\n        return cls(\n            rows=configuration.rows,\n            columns=configuration.columns,\n            x=configuration.inarow,\n            timeout=configuration.timeout,\n            steps=configuration.steps,\n        )\n\n    def to_configuration(self):\n        """\n        Creates a configuration from a game.\n\n        Returns:\n            kaggle_environments.utils.Struct: The configuration.\n        """\n        return Struct(\n            rows=self.rows,\n            columns=self.columns,\n            inarow=self.x,\n            timeout=self._timeout,\n            steps=self._steps,\n        )\n\n    def valid_actions(self, state):\n        return [\n            action\n            for action in self._actions\n            if not self._top & (1 << state.height[action])\n        ]\n\n    def do(self, state, action, inplace=False):\n        action_bit = 1 << state.height[action]\n        if not inplace:\n            state = deepcopy(state)\n        state.bitmaps[state.counter % 2] ^= action_bit\n        state.height[action] += 1\n        state.action_log.append(action)\n        state.counter += 1\n        return state\n\n    def undo(self, state, inplace=False):\n        action = state.action_log[-1]\n        action_bit = 1 << (state.height[action] - 1)\n        if not inplace:\n            state = deepcopy(state)\n        state.bitmaps[(state.counter - 1) % 2] ^= action_bit\n        state.height[action] -= 1\n        state.action_log.pop(-1)\n        state.counter -= 1\n        return state\n\n    def is_win(self, state, player):\n        directions = [1, 6, 7, 8]\n\n        # pylint: disable=no-member\n        def _is_win(bitmap, direction, x):\n            return np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)])\n\n        # pylint: enable=no-member\n\n        return any(\n            _is_win(state.bitmaps[player], direction, self.x)\n            for direction in directions\n        )\n\n    def is_draw(self, state):\n        return not self.valid_actions(state)\n\n    @property\n    def initial(self):\n        return self._STATE_CLS(bitmaps=[0, 0], mark=1,)\n\n    def __repr__(self):\n        return (\n            f"{self.__class__.__name__}(rows={self.rows}, "\n            f"columns={self.columns}, x={self.x})"\n        )\n',
    )
    __stickytape_write_module(
        "connect_x/game/game.py",
        '"""\nThis module represents a generic game and game state.\n"""\n\n\nclass GameState:\n    """\n    This class represent a abitrary state for a game.\n    """\n\n    @property\n    def state_hash(self):\n        """\n        Creates a unique hash for the state.\n        """\n        return None\n\n\nclass Game:\n    """\n    This class represent a abstract game.\n    """\n\n    _STATE_CLS = GameState\n\n    def valid_actions(self, state):\n        """\n        Return a list of the allowable moves at this point.\n        """\n        raise NotImplementedError\n\n    # pylint: disable=invalid-name\n    def do(self, state, action, inplace=False):\n        """\n        Return the state that results from making an action from a state.\n\n        Args:\n            state (GameState): The state.\n            action (int): The action.\n            inplace (boolean): If `False` returns a new game state. If `True`\n                overwrites the initial game state.\n\n        Returns:\n            GameState: The game state after performing the action.\n        """\n        raise NotImplementedError\n\n    # pylint: enable=invalid-name\n\n    def undo(self, state, inplace=False):\n        """\n        Returns the state that results from undoing the last action from a state.\n\n        Args:\n            state (GameState): The state.\n            inplace (boolean): If `False` returns a new game state. If `True`\n                overwrites the initial game state.\n\n        Returns:\n            GameState: The game state before performing the last action.\n        """\n        raise NotImplementedError\n\n    @property\n    def initial(self):\n        """\n        Returns the inital game state for this game.\n\n        Returns:\n            GameState: The initial game state.\n        """\n        raise NotImplementedError\n\n    def is_win(self, state, player):\n        """\n        Returns `True` if player has won in the state.\n\n        Args:\n            state (GameState): The state.\n            player (int): The player.\n\n        Returns:\n            boolean: Whether player has won in the state.\n        """\n        raise NotImplementedError\n\n    def is_draw(self, state):\n        """\n        Returns `True` if the game has ended in a draw.\n\n        Args:\n            state (GameState): The game state.\n\n        Returns:\n            boolean: Whether the game has ended in a draw.\n        """\n        raise NotImplementedError\n',
    )
    __stickytape_write_module(
        "connect_x/utils/converter.py",
        '"""\nThis module provides functions to represent boards as bitmaps, to check whether for\ntokens are connected and to make a move.\nThe code is taken from [Gilles Vandewiele\'s Medium Post]\n(https://towardsdatascience.com/\ncreating-the-perfect-connect-four-ai-bot-c165115557b0).\nAdjustment have been made based upon [Dominikus Herzberg\'s GitHub post]\n(https://github.com/denkspuren/BitboardC4/blob/master/BitboardDesign.md).\n"""\n\nimport numpy as np\n\n\ndef board_to_matrix(board, rows, columns):\n    """\n    Converts the board into a matrix.\n\n    Args:\n        board (list): Board provided by the `observation`.\n        rows (int): Number of rows, provided by the `configuration`.\n        columns (int): Number of columns, provided by the `configuration`.\n\n    Returns:\n        np.array: The board in matrix representation.\n    """\n    return np.array(board).reshape(rows, columns)\n\n\ndef _matrix_to_array(matrix):\n    matrix = np.insert(matrix, 0, 0, axis=0)\n    return np.rot90(matrix).reshape(1, -1)[0]\n\n\ndef _split_array(array):\n    def _bin_array(array, token):\n        assert token in (1, 2)\n        bin_array = np.zeros(array.shape)\n        bin_array[array == token] = 1\n        return bin_array.astype(int)\n\n    return _bin_array(array, 1), _bin_array(array, 2)\n\n\ndef _array_to_bitmap(array):\n    return int("".join(map(str, array)), 2)\n\n\ndef board_to_bitmaps(board, rows, columns):\n    """\n    Converts the board into a bitmap representation.\n\n    Args:\n        board (list): Board provided by the `observation`.\n        rows (int): Number of rows, provided by the `configuration`.\n        columns (int): Number of columns, provided by the `configuration`.\n\n    Returns:\n        list: List of two bitmaps (integers).\n    """\n    matrix = board_to_matrix(board, rows, columns)\n    array = _matrix_to_array(matrix)\n    array_me, array_other = _split_array(array)\n    return [_array_to_bitmap(array_me), _array_to_bitmap(array_other)]\n\n\ndef _bitmap_to_array(bitmap):\n    array = [int(bit) for bit in bin(bitmap)[2:]]\n    array = [0 for _ in range(49 - len(array))] + array\n    return np.array(array)\n\n\ndef _merge_arrays(array_me, array_other):\n    assert array_me.shape == array_other.shape\n    array = np.zeros(array_me.shape)\n\n    array[array_me == 1] = 1\n    array[array_other == 1] = 2\n\n    return array.astype(int)\n\n\ndef _array_to_matrix(array):\n    matrix = array.reshape(7, 7)[:, 1:]\n    return np.rot90(matrix, k=3)\n\n\ndef matrix_to_board(matrix):\n    """\n    Converts the matrix into a board.\n\n    Args:\n        matrix (np.array): The board in matrix representation.\n\n    Returns:\n        list: The board.\n    """\n    return matrix.reshape(1, -1).tolist()[0]\n\n\ndef bitmaps_to_board(bitmaps):\n    """\n    Converts the bitmaps into a board.\n\n    Args:\n        bitmaps (list): List of two bitmaps (integers).\n\n    Returns:\n        list: The board.\n    """\n    matrix = bitmaps_to_matrix(bitmaps)\n    board = matrix_to_board(matrix)\n    return board\n\n\ndef bitmaps_to_matrix(bitmaps):\n    """\n    Converts the bitmaps into a matrix.\n\n    Args:\n        bitmaps (list): List of two bitmaps (integers).\n\n    Returns:\n        np.array: The board in matrix representation.\n    """\n    bitmap_me, bitmap_other = bitmaps\n    array_me = _bitmap_to_array(bitmap_me)\n    array_other = _bitmap_to_array(bitmap_other)\n    array = _merge_arrays(array_me, array_other)\n    matrix = _array_to_matrix(array)\n    return matrix\n',
    )
    __stickytape_write_module("connect_x/utils/__init__.py", "")
    __stickytape_write_module(
        "connect_x/action_catalogue.py",
        '"""\nThis module contains hard-coded rules for Connect X.\n"""\n\nimport numpy as np\n\nfrom .utils.converter import bitmaps_to_matrix\n\n\ndef _middle_column(matrix):\n    return np.floor(matrix.shape[1] / 2).astype(int)\n\n\n# pylint: disable=unused-argument\ndef _action_0(matrix, state):\n    return _middle_column(matrix)\n\n\ndef _action_1(matrix, state):\n    middle_column = _middle_column(matrix)\n    if matrix[-1, middle_column] != 0:\n        return middle_column + 1\n    return middle_column\n\n\ndef _action_2(matrix, state):\n    middle_column = _middle_column(matrix)\n    if (\n        matrix[-1, middle_column] == state.mark\n        and matrix[-1, middle_column + 1] == 0\n        and matrix[-1, middle_column - 1] == 0\n        and matrix[-1, middle_column - 2] == 0\n        and matrix[-1, middle_column + 2] == 0\n    ):\n        return middle_column + 1\n    return None\n\n\n# pylint: enable=unused-argument\n\n\n_ACTION_CATALOGUE = {\n    0: _action_0,\n    1: _action_1,\n    2: _action_2,\n}\n\n\ndef get_action(state):\n    """\n    Returns an action from the _ACTION_CATALOGUE, given the state.\n\n    Args:\n        state (connect_x.game.connect_x.ConnectXState): The state.\n\n    Returns:\n        int: The action.\n    """\n    matrix = bitmaps_to_matrix(state.bitmaps)\n    action_func = _ACTION_CATALOGUE.get(state.counter)\n    if action_func:\n        return action_func(matrix, state)\n    return None\n',
    )
    __stickytape_write_module(
        "connect_x/minimax.py",
        '"""\nThis module implements the Minimax search algorithm with Alpha-Beta pruning.\nThe implementation is taken from here:\nhttps://github.com/aimacode/aima-python/blob/master/games.py\n"""\n\nimport functools\n\nimport numpy as np\n\n\nclass StateValueCache:\n    """\n    A cache that stores the value for every state.\n    """\n\n    def __init__(self, func, cache=None):\n        functools.update_wrapper(self, func)\n        self.func = func\n        self.cache = cache or {}\n        self.cache_calls = 0\n\n    def _cached_value(self, key):\n        if key is not None:\n            value = self.cache.get(key)\n            if value is not None:\n                self.cache_calls += 1\n            return value\n        return None\n\n    def _cache_value(self, key, value):\n        if key is not None:\n            self.cache[key] = value\n\n    def reset_cache(self):\n        """\n        Resets the cache.\n        """\n        self.cache = {}\n\n    def __call__(self, *args, **kwargs):\n        key = kwargs.get("state").state_hash\n        value = self._cached_value(key) or self.func(*args, **kwargs)\n        self._cache_value(key, value)\n        return value\n\n\ndef is_terminated(state, game, player):\n    """\n    Returns the value of the game state and whether the game has terminated.\n\n    Args:\n        state (connect_x.game.connect_x.ConnectXState): The state.\n        game (connect_x.game.connect_x.ConnectXGame): The game.\n        player (int): The player (1 or 0).\n\n    Returns:\n        tuple: The value of the state for the player and whether the game has ended.\n    """\n    if game.is_win(state, player):\n        return np.inf, True\n    if game.is_win(state, 1 - player):\n        return -np.inf, True\n    if game.is_draw(state):\n        return 0, True\n    return None, False\n\n\ndef negamax(\n    game,\n    state,\n    depth,\n    player,\n    heuristic_func,\n    order_actions_func=None,\n    return_cache=False,\n):\n    """\n    Applies the Negamax algorithm to the game to determine the next best action for\n    the player, given the state.\n\n    Args:\n        game (connect_x.game.connect_x.ConnectXGame): The game.\n        state (connect_x.game.connect_x.ConnectXState): The state.\n        depth (int): The maximum depth of the Negamax tree.\n        player (int): The player (1 or 0).\n        heuristic_func (callable, optional): The heuristic function for the state when\n            the tree can not be resolved up to a terminal leaf.\n        order_actions_func (callable, optional): The function that determines in which\n            order the actions will be evaluated.\n        return_cache (bool, optional): Whether to return the entire cache, instead of\n            just the value.\n\n    Returns:\n        float: When `return_cache=False`.\n        dict: When `return_cache=True`.\n    """\n    heuristic_func = heuristic_func or (lambda state, player: 0)\n    order_actions_func = order_actions_func or (lambda actions: actions)\n    alpha = -np.inf\n    beta = np.inf\n\n    @StateValueCache\n    def _negamax(state, game, depth, alpha, beta, maximize):\n        value, terminated = is_terminated(state, game, player)\n        if terminated:\n            return value * maximize\n        if depth == 0:\n            value = heuristic_func(state, player) * maximize\n            return value\n\n        actions = game.valid_actions(state)\n        actions = order_actions_func(actions)\n\n        value = -np.inf\n\n        for action in actions:\n            value = max(\n                value,\n                -_negamax(\n                    state=game.do(state, action),\n                    game=game,\n                    depth=depth - 1,\n                    alpha=-beta,\n                    beta=-alpha,\n                    maximize=-maximize,\n                ),\n            )\n            alpha = max(alpha, value)\n            if alpha >= beta:\n                break\n\n        return value\n\n    _negamax.reset_cache()\n    value = _negamax(\n        state=state, game=game, depth=depth, alpha=alpha, beta=beta, maximize=1\n    )\n\n    if return_cache:\n        return _negamax\n    return value\n',
    )
    __stickytape_write_module(
        "connect_x/config.py",
        '"""\nThis module describes the config for the main submission file.\n"""\n\nimport numpy as np\n\n\ndef connected(state, player, x):\n    """\n    Returns how many times x are connected in the state of player.\n    """\n    directions = [1, 6, 7, 8]\n\n    def _connected(bitmap, direction, x):\n        assert x > 0\n        # pylint: disable=no-member\n        return bin(np.bitwise_and.reduce([bitmap >> i * direction for i in range(x)]))[\n            2:\n        ].count("1")\n        # pylint: enable=no-member\n\n    return np.array(\n        [_connected(state.bitmaps[player], direction, x) for direction in directions]\n    ).sum()\n\n\ndef heuristic(state, player):\n    """\n    The heuristic function.\n\n    Args:\n        state (connect_x.game.connect_x.ConnectXState): The state.\n        player (int): The player (0 or 1).\n\n    Returns:\n        float: The heuristic value.\n    """\n\n    def _heuristic_player(player):\n        return np.sum([4 ** (x - 1) * connected(state, player, x) for x in range(2, 4)])\n\n    return _heuristic_player(player) - _heuristic_player(1 - player)\n\n\ndef order_actions(actions):\n    """\n    Orders the actions for the Negamax tree.\n\n    Args:\n        actions (list): The list of actions.\n\n    Returns:\n        list: The list of prioritized actions.\n    """\n    order = np.array(actions)\n    order = np.absolute(order - 3)\n    order = np.array(np.argsort(order))\n    return np.array(actions)[order]\n\n\nDEPTH = 5\n',
    )
    __stickytape_write_module(
        "connect_x/utils/logger.py",
        '"""\nProvides a utils method to setup the logger.\n"""\nimport os\nimport logging\n\n\ndef setup_logger(logger_name=None):\n    """\n    Get the logger.\n\n    Args:\n        logger_name (str): The name of the logger.\n    Returns:\n        logging.Logger: The logger.\n    """\n    logger = logging.getLogger(logger_name)\n    handler = logging.StreamHandler()\n    formatter = logging.Formatter(\n        "%(levelname)s - %(name)s - %(asctime)s - %(message)s",\n        datefmt="%m/%d/%Y %I:%M%S %p",\n    )\n    handler.setFormatter(formatter)\n    logger.addHandler(handler)\n    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))\n    return logger\n',
    )
    """
    This module contains the submission for the Kaggle competition.
    """

    from datetime import datetime

    import numpy as np

    from connect_x.game import ConnectXGame, ConnectXState
    from connect_x.action_catalogue import get_action
    from connect_x.minimax import negamax
    from connect_x.config import heuristic, order_actions, DEPTH

    from connect_x.utils.logger import setup_logger

    _LOGGER = setup_logger(__name__)

    def _catalogued_action(state):
        return get_action(state)

    def _planned_action(game, state):
        cache = negamax(
            game=game,
            state=state,
            depth=DEPTH,
            heuristic_func=heuristic,
            order_actions_func=order_actions,
            player=state.mark - 1,
            return_cache=True,
        )
        valid_actions = order_actions(game.valid_actions(state))
        valid_states = [game.do(state, action) for action in valid_actions]
        values = [
            cache.cache.get(state.state_hash, np.inf) * -1 for state in valid_states
        ]
        _LOGGER.debug(zip(valid_actions, values))
        return valid_actions[np.argmax(values)]

    def act(observation, configuration):
        """
        Decides what action to do next.
    
        Args:
            observation (kaggle_environments.utils.Struct): The observation.
            configuration (kaggle_environments.utils.Struct): The configuration.
    
        Returns:
            int: The action.
        """
        start = datetime.now()
        game = ConnectXGame.from_configuration(configuration)
        state = ConnectXState.from_observation(
            observation, configuration.rows, configuration.columns
        )
        action = _catalogued_action(state) or _planned_action(game, state)
        end = datetime.now()
        _LOGGER.info(f"Action selected: '{action}'.")
        _LOGGER.debug(f"Time taken: {end - start}.")

        return int(action)

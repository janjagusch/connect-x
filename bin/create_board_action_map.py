"""
This module creates the `connect_x/board_action_map` module.
"""

from datetime import datetime
import gzip
import json

from kaggle_environments import make
import numpy as np

from connect_x.minimax import minimax, ConnectXNode
from connect_x.utils import board as board_utils
from connect_x.utils.logger import setup_logger
from submission import FORECAST_DEPTH


_LOGGER = setup_logger(__name__)


def flatten_tree(node):
    """
    Flatten a tree of ConnectXNodes into a dictionary.

    Args:
        node (connect_x.minimax.tree.ConnectXNode): The root node.

    Returns:
        dict: The flat tree as a dictionary.
    """
    key = board_utils.board_hash(node.observation.board, node.observation.mark)
    flat_tree = {key: node}
    for child in node.children:
        flat_tree = {**flat_tree, **flatten_tree(child)}
    return flat_tree


def best_action(node):
    """
    Returns the best action for a maximize node.

    Args:
        node (connect_x.minimax.tree.ConnectXNode): The root node.

    Returns:
        int: The action.
    """
    assert node.maximize
    if not node.children:
        return None
    return int(
        node.children[np.argmax([child.value[0] for child in node.children])].action
    )


def write_module(board_action_map, forecast_depth):
    """
    Writes the `connect_x/board_action_map` module.

    Args:
        board_action_map (dict): The board action map.
        forecast_depth (int): How many turns were forecasted to produce the actions.
    """
    board_action_map_binary = gzip.compress(
        json.dumps(board_action_map).encode("ascii")
    )
    with open("connect_x/board_action_map.py", "w") as file_pointer:
        file_pointer.write(
            '"""\nThis module contains pre-calculated board actions.\n"""\n'
        )
        file_pointer.write("import gzip\nimport json\n\n")
        file_pointer.write(f"FORECAST_DEPTH = {forecast_depth}\n\n")
        file_pointer.write("# pylint: disable=line-too-long\n")
        file_pointer.write(f"_BOARD_ACTION_MAP_BINARY = {board_action_map_binary}\n\n")
        file_pointer.write("# pylint: enable=line-too-long\n")
        file_pointer.write(
            "BOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))\n"
        )


def main(forecast_depth):
    """
    Calcuates the Minimax trees, generated the board-action map and writes it into
    a module.

    Args:
        forecast_depth (int): How many turns should be forecasted in the Minimax.
    """
    env = make("connectx")
    observation = env.state[0].observation
    configuration = env.configuration

    node_1 = ConnectXNode(observation, configuration)
    node_2 = ConnectXNode(observation, configuration, mark=1)

    _LOGGER.info(f"Starting Minimax with max_depth={forecast_depth}...")
    start = datetime.now()
    minimax(node_1, max_depth=forecast_depth)
    minimax(node_2, max_depth=forecast_depth, maximize=False)
    end = datetime.now()
    _LOGGER.info(f"Minimax completed. Time taken: {end - start}.")

    _LOGGER.info("Flattening Minimax tree ...")
    start = datetime.now()
    flat_tree_1 = flatten_tree(node_1)
    flat_tree_2 = flatten_tree(node_2)
    end = datetime.now()
    _LOGGER.info(f"Flattening completed. Time taken: {end - start}.")
    flat_tree_1 = {key: value for key, value in flat_tree_1.items() if value.maximize}
    flat_tree_2 = {key: value for key, value in flat_tree_2.items() if value.maximize}
    flat_tree = {**flat_tree_1, **flat_tree_2}
    flat_tree = {
        key: value
        for key, value in flat_tree.items()
        if value.depth < (forecast_depth - FORECAST_DEPTH)
    }
    flat_tree = {key: best_action(value) for key, value in flat_tree.items()}
    _LOGGER.info(f"Flat tree has {len(flat_tree)} entries.")

    _LOGGER.info("Writing dictionary into module ...")
    write_module(flat_tree, forecast_depth)
    _LOGGER.info("Writing completed.")


if __name__ == "__main__":

    main(7)

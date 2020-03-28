# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python (connect-x)
#     language: python
#     name: connect-x
# ---

from datetime import datetime
import gzip
import json

from kaggle_environments import make
import numpy as np

from connect_x.minimax import minimax, ConnectXNode

7 ** 8

env = make("connectx")

observation = env.state[0].observation
configuration = env.configuration

node_1 = ConnectXNode(observation, configuration)
node_2 = ConnectXNode(observation, configuration, mark=1)

# +
MAX_DEPTH = 7
POSSIBLE_FORECAST = 3

start = datetime.now()
minimax(node_1, max_depth=MAX_DEPTH)
minimax(node_2, max_depth=MAX_DEPTH, maximize=False)
end = datetime.now()

print(f"time taken: {end - start}.")


# -


def make_board_player_agnostic(board, mark):
    def agnostic(val, mark):
        if val == 0:
            return 0
        return 1 if val == mark else 9

    return [agnostic(i, mark) for i in board]


def flatten_tree(node, flat_tree=None):
    flat_tree = flat_tree or {}
    key = "".join(
        str(i)
        for i in make_board_player_agnostic(
            node.observation.board, node.observation.mark
        )
    )
    flat_tree[key] = node
    for child in node.children:
        flatten_tree(child, flat_tree)
    return flat_tree


flat_tree_1 = flatten_tree(node_1)
flat_tree_2 = flatten_tree(node_2)

flat_tree_1 = {key: value for key, value in flat_tree_1.items() if value.maximize}
flat_tree_2 = {key: value for key, value in flat_tree_2.items() if value.maximize}

list(flat_tree_1.keys())[:5]

list(flat_tree_2.keys())[:5]

len(flat_tree_1)

len(flat_tree_2)

flat_tree = {**flat_tree_1, **flat_tree_2}

flat_tree = {
    key: value
    for key, value in flat_tree.items()
    if value.depth < (MAX_DEPTH - POSSIBLE_FORECAST)
}

len(flat_tree)


def best_action(node):
    assert node.maximize
    if not node.children:
        return None
    return node.children[np.argmax([child.value[0] for child in node.children])].action


flat_tree = {key: best_action(value) for key, value in flat_tree.items()}

len(flat_tree)

flat_tree = {key: int(value) for key, value in flat_tree.items()}

flat_tree["000000000000000000000000000000000000001900"]

board_value_map_binary = gzip.compress(json.dumps(flat_tree).encode("ascii"))
with open("../connect_x/board_value_map.py", "w") as file_pointer:
    file_pointer.write("import gzip\nimport json\n\n")
    file_pointer.write(f"FORECAST_DEPTH = {MAX_DEPTH}\n\n")
    file_pointer.write(f"_BOARD_VALUE_MAP_BINARY = {board_value_map_binary}\n\n")
    file_pointer.write(
        "BOARD_VALUE_MAP = json.loads(gzip.decompress(_BOARD_VALUE_MAP_BINARY))\n"
    )

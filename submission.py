import gzip

from connect_x.minimax import minimax, ConnectXNode
from connect_x.move_catalogue import move
from connect_x.board_value_map import BOARD_VALUE_MAP, FORECAST_DEPTH


def act(observation, configuration):
    action = move(observation, configuration)
    if action is None:
        node = ConnectXNode(observation, configuration)
        next_node, value = minimax(node, max_depth=3)
        action = next_node.action
    return int(action)

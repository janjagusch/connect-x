import gzip

from connect_x.minimax import minimax, ConnectXNode
from connect_x.move_catalogue import move
from connect_x.state_value import BOARD_VALUE_MAP


def act(observation, configuration):
    action = move(observation, configuration)
    if action is None:
        node = ConnectXNode(observation, configuration)
        next_node, value = minimax(node, max_depth=3)
        action = next_node.action
    return int(action)

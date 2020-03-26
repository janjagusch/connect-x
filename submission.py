from connect_x.tree import ConnectXNode
from connect_x.minimax import minimax


def act(observation, configuration):
    node = ConnectXNode(observation, configuration)
    next_node, value = minimax(node, max_depth=4)
    return int(next_node.action)

from connect_x.minimax import minimax, ConnectXNode


def act(observation, configuration):
    node = ConnectXNode(observation, configuration)
    next_node, value = minimax(node, max_depth=3)
    return int(next_node.action)

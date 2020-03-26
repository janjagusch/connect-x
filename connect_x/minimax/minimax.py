"""
This module implementy the Minimax algorithm.
"""

import numpy as np


def minimax(node, max_depth=4, maximize=True, current_depth=None):
    """
    Executes the Minimax algorithm.

    Args:
        node (connect_x.minimax.tree.ConnectXNode): The root node.
        max_depth (int, optional): The maximum recursion depth.
        maximize (bool, optional): Whether to maximize or minimize.
        current_depth (int, optional): The current depth.

    Returns:
        tuple: (Next node to go for, Value of the next node).
    """
    current_depth = current_depth or 0
    value, terminated = node.value
    if (current_depth == max_depth) or terminated:
        return value
    children = node.make_children()
    values = [
        minimax(child, max_depth, not maximize, current_depth + 1) for child in children
    ]
    if maximize:
        value = np.max(values)
        index = np.argmax(values)
    else:
        value = np.min(values)
        index = np.argmin(values)
    if not current_depth:
        return children[index], value
    return value

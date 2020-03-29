"""
This module provides a bitmap based simulator.
"""

_ACTIONS = range(7)
_TOP = int("_".join("1000000" for _ in range(7)), 2)


class Simulator:
    def __init__(self, bitmaps, action_log=None, height=None, counter=None):
        self.bitmaps = bitmaps
        self._action_log = action_log or []
        self._height = height or [action * 7 for action in _ACTIONS]
        self._counter = counter or 0

    @property
    def action_log(self):
        return self._action_log

    @property
    def height(self):
        return self._height

    @property
    def counter(self):
        return self._counter

    @property
    def valid_actions(self):
        return [action for action in _ACTIONS if not _TOP & (1 << self.height[action])]

    def do_action(self, action):
        action_bit = 1 << self._height[action]
        self.bitmaps[self._counter % 2] ^= action_bit

        self._height[action] += 1
        self._action_log.append(action)
        self._counter += 1

    def undo_action(self):
        action = self._action_log[-1]
        action_bit = 1 << (self.height[action] - 1)
        self.bitmaps[(self._counter - 1) % 2] ^= action_bit

        self._height[action] -= 1
        self._action_log.pop(-1)
        self._counter -= 1


def is_win(bitmap):
    directions = [1, 6, 7, 8]

    def _is_win(bitmap, direction):
        bb = bitmap & (bitmap >> direction)
        return bb & (bb >> 2 * direction)

    return any(_is_win(bitmap, direction) for direction in directions)

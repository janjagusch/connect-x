"""
This module implements an evaluator class for Connect-X game states.
"""

import itertools

import numpy as np

from connect_x.utils.singleton import Singleton


def _top(rows, columns):
    def __top(rows):
        return "".join(["1", "".join("0" for _ in range(rows))])

    return int("".join(__top(rows) for _ in range(columns)), 2)


def _length(rows, columns):

    return int("".join("1" for _ in range(rows * columns)), 2)


class ConnectXStateEvaluator:

    _SHIFT_OPERATORS = [np.left_shift, np.right_shift]

    def __init__(self, rows=6, columns=7, window=4):
        self.rows = rows
        self.columns = columns
        self.window = window
        self._directions = [1, columns - 1, columns, columns + 1]
        self._top = _top(rows, columns)
        self._length = _length(rows + 1, columns)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rows={self.rows}, "
            f"columns={self.columns}, window={self.window})"
        )

    def _available_cells(self, bitmap):
        return self._length & ~bitmap & ~self._top

    def _connected_cells(self, bitmap, shift_operator, direction, window):
        return np.bitwise_and.reduce(
            [
                self._length,
                *[shift_operator(bitmap, direction * i) for i in range(window)],
            ]
        )

    @staticmethod
    def _count_bits(bitmap):
        return bin(bitmap).split("b", 1)[-1].count("1")

    def __evaluate(self, bitmaps, window, shift_operator, direction):
        available_connected = self._connected_cells(
            self._available_cells(bitmaps[1]), shift_operator, direction, self.window
        )
        occupied_connected = self._connected_cells(
            bitmaps[0], shift_operator, direction, window
        )
        return self._count_bits(available_connected & occupied_connected & self._length)

    def _evaluate(self, bitmaps):
        value = 0
        for window in range(1, self.window):
            val = 0
            for args in itertools.product(self._SHIFT_OPERATORS, self._directions):
                val += self.__evaluate(bitmaps, window, *args)
            val = window ** val
            value += val
        return value

    def __call__(self, state):
        return self._evaluate(state.bitmaps) - self._evaluate(state.bitmaps[::-1])

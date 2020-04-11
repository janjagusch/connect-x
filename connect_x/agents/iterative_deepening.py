"""
This module provides an iterative deepening class that can be used as a decorator.
"""

import asyncio
import functools

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


class IterativeDeepening:
    """
    Applies iterative deepening to a function. Terminates when maximum depth is reached
    or because of a timeout.
    Use it as a decorator.

    Args:
        func (callable): A function that returns a result.
        arg (string, optional): The name of the argument you want to iteratively
            deepen.
        timeout (int): The maximum time (in seconds) for iterative deepening to run.
        min_depth (int, optional): The minimum depth for iterative deepening.
        max_depth (int, optional): The maximum depth for iterative deepening.
    """

    def __init__(self, func, arg="depth", timeout=1, min_depth=1, max_depth=None):
        functools.update_wrapper(self, func)
        self.func = func
        self.arg = arg
        self.timeout = timeout
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.result = None

    def __call__(self, *args, **kwargs):
        self.result = None

        async def call_with_timeout():
            try:
                await asyncio.wait_for(
                    self.__iterative_deepening(*args, **kwargs), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                _LOGGER.debug(f"Timed out internally")
                return

        asyncio.run(call_with_timeout())
        return self.result

    async def __iterative_deepening(self, *args, **kwargs):
        """
        Repeats the minimax algorithm with an increasingle larger depth, and
        saves the latest result to a nonlocal variable in the closure.
        """
        for depth in range(self.min_depth, self.max_depth + 1):
            _LOGGER.debug(f"Starting minimax with depth {depth}")
            self.result = await self.func(*args, **{**kwargs, self.arg: depth})
            _LOGGER.debug(f"Minimax with depth {depth} yielded action: {self.result}")

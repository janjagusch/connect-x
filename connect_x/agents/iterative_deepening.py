"""
This module provides an iterative deepening class that can be used as a decorator.
"""

import functools
import time
from multiprocessing import Process, Value

from connect_x.utils.logger import setup_logger


_LOGGER = setup_logger(__name__)


class IterativeDeepening:
    """
    Applies iterative deepening to a function. Terminates when maximum depth is reached
    or because of a timeout.
    Use it as a decorator.

    Args:
        func (callable): A function with an argument `depth` that returns a result.
        arg (string, optional): The name of the argument you want to iteratively
            deepen.
        timeout (int): The maximum time (in seconds) for iterative deepening to run.
        min_depth (int, optional): The minimum depth for iterative deepening.
        max_depth (int, optional): The maximum depth for iterative deepening.
    """

    def __init__(self, func, arg="depth", timeout=1, min_depth=0, max_depth=None):
        functools.update_wrapper(self, func)
        self.func = func
        self.arg = arg
        self.timeout = timeout
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __run_func(self, result, *args, **kwargs):
        result.value = self.func(*args, **kwargs)

    def _run_func(self, start_time, depth, *args, **kwargs):
        result = Value("i", 1)
        func_process = Process(
            target=self.__run_func,
            args=[result, *args],
            kwargs={"depth": depth, **kwargs},
        )
        func_process.start()
        while time.perf_counter() - start_time <= self.timeout:
            if not func_process.is_alive():
                break
            time.sleep(0.001)
        else:
            _LOGGER.debug(
                f"Timed out! Total time taken: {time.perf_counter() - start_time}."
            )
            func_process.terminate()
            func_process.join()
            raise TimeoutError
        return result.value

    def __call__(self, *args, **kwargs):
        depth = self.min_depth
        result = None
        start_time = time.perf_counter()

        while True:
            if self.max_depth and depth > self.max_depth:
                _LOGGER.debug(
                    f"Maximum depth={self.max_depth} reached. "
                    "Breaking out of the loop."
                )
                break
            _LOGGER.debug(f"Starting iterative deepening with depth={depth}.")
            try:
                result = self._run_func(start_time, depth, *args, **kwargs)
            except TimeoutError:
                break
            _LOGGER.debug(
                f"Iterative deepening with depth={depth} completed. "
                f"Result: {result}"
            )
            depth += 1

        return result

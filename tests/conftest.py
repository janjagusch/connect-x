"""
The conftest file.
"""

import pytest
import numpy as np
from kaggle_environments import make


@pytest.fixture(name="to_array")
def to_array_():
    """
    Converts list to array.
    """
    # pylint: disable=unnecessary-lambda
    return lambda x: np.array(x)
    # pylint: enable=unnecessary-lambda


@pytest.fixture(name="env", scope="function")
def env_():
    """
    Connectx environment.
    """
    return make("connectx", debug=True)


@pytest.fixture(name="configuration", scope="function")
def configuration_(env):
    """
    Connectx configuration.
    """
    return env.configuration


@pytest.fixture(name="observation", scope="function")
def observation_(env):
    """
    Connectx observation.
    """
    return env.state[0].observation

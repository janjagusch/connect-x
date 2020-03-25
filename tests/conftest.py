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

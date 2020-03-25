"""
The conftest file.
"""

import pytest
import numpy as np
from kaggle_environments import make


@pytest.fixture(name="to_array")
def to_array_():
    return lambda x: np.array(x)


@pytest.fixture(name="env", scope="function")
def env_():
    return make("connectx", debug=True)

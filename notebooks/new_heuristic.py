# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python (connect-x)
#     language: python
#     name: connect-x
# ---

import random
import sys

sys.path.append("..")

from kaggle_environments import make
from kaggle_environments.core import Environment

from submission_standalone import act

env = make("connectx")


def random_state(env=None, steps=6):
    env = env or make("connectx")
    env.reset()
    for i in range(steps):
        action = random.choice(range(env.configuration.columns))
        env.step([action, action])
    return env.state


observation = env.state[0].observation
configuration = env.configuration

act(observation, configuration)

env.run([act, "negamax"], state=random_state(env, steps=0))
env.render(mode="ipython", width=500, height=450)

env.play(agents=[None, act])


import numpy as np

state = random_state(steps=12)

from connect_x import utils

matrix = utils.board_to_matrix(
    state[0].observation.board, env.configuration.rows, env.configuration.columns
)

matrix


n_tokens_per_column = (matrix != 0).sum(axis=0)

cell_height = np.indices(matrix.shape)[0][::-1] + 1

distance_penalty = cell_height - n_tokens_per_column
distance_penalty[distance_penalty < 1] = 1
distance_penalty

window = 4
mark = 1


def horizontal_windows(matrix, row_index, col_index, window_size):
    n_rows, n_cols = matrix.shape
    start_col_index = np.max([0, col_index - (window_size - 1)])
    end_col_index = np.min([col_index, n_cols - (window_size - 1)])
    return np.array(
        [
            matrix[row_index, i : i + window_size]
            for i in range(start_col_index, end_col_index)
        ]
    )


def vertical_windows(matrix, row_index, col_index, window_size):
    return horizontal_windows(
        matrix=matrix.T,
        row_index=col_index,
        col_index=row_index,
        window_size=window_size,
    )


def diagonal_windows(matrix, row_index, col_index, window_size):
    # TODO...
    raise NotImplementedError


matrix

horizontal_windows(matrix, row_index=4, col_index=4, window_size=4)

vertical_windows(matrix, row_index=4, col_index=4, window_size=4)

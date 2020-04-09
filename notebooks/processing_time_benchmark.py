# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Processing Time Benchmark

# In this notebook, you can benchmark the processing time of your algorithm, given 6,000 different test states from [gamesolver.org](http://blog.gamesolver.org/solving-connect-four/02-test-protocol/).
#
# To run this notebook, make sure that the `/data/test_sets/` directory contains six files. If not, please execute `bin/request_test_sets`.

import sys

sys.path.append("..")

import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm


from connect_x.game.connect_x import ConnectXGame
from connect_x.agents import negamax
from connect_x.config import order_actions, heuristic

tqdm.pandas()


def read_test_set(file_path):
    # Reads a test set from a file path.
    test_set = pd.read_csv(file_path, sep=" ", header=None, names=["actions", "value"])
    test_set["name"] = file_path.split("/")[-1]
    return test_set


# +
# Loading the test sets.

DIR = "../data/test_sets"

test_set = pd.concat(
    [read_test_set(os.path.join(DIR, test_set)) for test_set in os.listdir(DIR)]
)
test_set["actions"] = test_set["actions"].astype(str)
test_set["n_actions"] = test_set["actions"].str.len()
test_set = test_set.sort_values("n_actions").reset_index(drop=True)

test_set


# -


def create_state(actions, state=None):
    # Creates a state from a sequence of actions.
    game = ConnectXGame()
    if not state:
        state = game.initial
    for action in actions:
        # Indexed at 1, pathetic.
        game.do(state, int(action) - 1, inplace=True)
    return state


test_set["state"] = test_set["actions"].apply(create_state)


def negamax_func(depth=10):
    def _negamax(state):
        # Runs the negamax returns the  value and how long it took to process.
        game = ConnectXGame()
        start = datetime.now()
        value = negamax(
            game=game, state=state, depth=depth, player=1, order_actions_func=order_actions, heuristic_func=heuristic
        )
        end = datetime.now()
        return pd.Series(
            {
                "minimax_value": value,
                "minimax_duration_seconds": (end - start).total_seconds(),
            }
        )
    return _negamax


# +
# You can adjust the range of number of actions that you want to benchmark here.
# If you go much lower, please also adjust `depth` in `run_negmax`.

test_set_filtered = test_set[test_set["n_actions"] >= 22]
test_set_filtered

# +
# Calculates the benchmark results. For `n_actions>=22`, this took 8:30 minutes for me.

test_set_filtered = test_set_filtered.merge(
    test_set_filtered["state"].progress_apply(negamax_func(depth=5)),
    left_index=True,
    right_index=True,
)

# +
ax = (
    test_set_filtered.groupby("n_actions")["minimax_duration_seconds"]
    .quantile([0.9, 0.95, 0.99, 1])
    .unstack()
    .plot(
        xticks=range(
            test_set_filtered["n_actions"].min(),
            test_set_filtered["n_actions"].max(),
            1,
        ),
        figsize=(16, 9),
        grid=True,
    )
)

_ = ax.set_xlabel("Number of actions")
_ = ax.set_ylabel("Processing time in seconds")
# -







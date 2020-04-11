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

# # Benchmark Environment

# This notebook provides a benchmark environment, where you can compare two versions of your agents and let them compete against each other.
#
# **Requirements:**
# * You have two standalone submissions in your `/benchmark` directory (`agent_1_standalone.py` and `agent_2_standalone.py`)
# * You have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your machine.
#
# **Getting Started:**
# * Navigate to `/benchmark`
# * Execute `docker-compose up --build`
# * Then start this notebook.
#
# **Further information:**
# * You can also let your algorithm compete against `random` or `negamax` agent.
# * You will need [Jupytext](https://github.com/mwouts/jupytext) to convert this `.py` light script into a Jupyter notebook.

from kaggle_environments import make
import requests

import sys

sys.path.append("..")
from submission import act as current_act

env = make("connectx", debug=True)
_ = env.reset()


def act(url, observation, configuration):
    """
    Sends a post request to one of the two agents.
    """
    data = {"observation": observation, "configuration": configuration}
    response = requests.post(url=url, json=data).json()
    return response["best_action"]


act_1 = lambda observation, configuration: act(
    "http://localhost:8081/actions/best_action", observation, configuration
)
act_2 = lambda observation, configuration: act(
    "http://localhost:8082/actions/best_action", observation, configuration
)

# Replace `act_1` or `act_2` with "random" or "negamax" to play against default agents.
env.run(["negamax", act_1])
env.render(mode="ipython", width=500, height=450)

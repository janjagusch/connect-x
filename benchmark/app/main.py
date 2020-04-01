"""
This module serves the agent through a Flask web API.
"""

from flask import Flask
from flask import request
from kaggle_environments.utils import Struct

from agent import act


app = Flask(__name__)


@app.route("/actions/best_action", methods=["POST"])
def best_action():

    observation = Struct(**request.json["observation"])
    configuration = Struct(**request.json["configuration"])

    return {"best_action": act(observation, configuration)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

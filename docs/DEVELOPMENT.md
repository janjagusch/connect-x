# Development

This document serves to onboard new developers to this project.

## Getting Started

### Requirements

* You have stored the Kaggle API credentials file according the [official instructions](https://github.com/Kaggle/kaggle-api#api-credentials).
* You have [pyenv](https://github.com/pyenv/pyenv) installed on your machine.
* You have [Poetry](https://github.com/python-poetry/poetry) installed in your global Python environment.

### Installation

1. Make sure the directory is pointing to a compatible Python version.
    ```shell
    pyenv local 3.7.5 3.8.1 # Example
    ```

1. Clone the repository from GitHub.

    ```shell
    git clone git@github.com:janjagusch/connect-x.git
    ```

1. Install the dependencies.

    ```shell
    poetry install
    ```

### Validation

After the installation, make sure that

* ... all code is formatted correctly:
    ```shell
    make lint
    ```

* ... all tests are passed:
    ```shell
    make test
    ```

* ... the `submission_standalone.py` is valid:
    ```shell
    make validate_submission
    ```

## Project Structure

This section introduces you to the project structure.

### The `submission_standalone.py` File

The most important file in this project is the `submission_standalone.py` script, which gets submitted to Kaggle. The reason is that Kaggle only allows for single file submissions, which is why we have to consolidate the entire project into one Python module, using [stickytape](https://github.com/mwilliamson/stickytape). Whenever you make changes to the project or the `submission.py` file, make sure to run:
    
```shell
make standalone_submission && make validate_submission
```

### The `connect_x` Package

Currently divided into three sections:

* `game/`: Contains the general logic for Connect-X, such as
    * Creating game and state objects.
    * Determining whether a player has won in a state.
    * Determining valid actions from a state.
    * Applying an action to a state and receiving a new state.
* `agents/`: Contains all necessary logic to build somehow intelligent agents, such as:
    * The [Negamax](https://en.wikipedia.org/wiki/Negamax) algorithm.
    * [Iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search).
    * State-value caching.
* `utils/`: Contains all helper logic, such as:
    * Converting boards into matrices and bitmaps.
    * Setting up a logger.

## CI/CD

We use [Travis](https://travis-ci.com/github/janjagusch/connect-x). Make sure you have access to it, otherwise request it from [Jan](jan.jagusch@gmail.com). Everytime you push a tag to the repository the `submission_standalone` will be submitted as a solution to Kaggle.

## Bitmap Representation

We encode the board state as two bitmaps - one for each player - where `1`s indicate that there is a token in that cell in the board. For more information, please refer to this [explanation post](https://github.com/denkspuren/BitboardC4/blob/master/BitboardDesign.md).

## Benchmarking

### Game Simulation

This section demonstrates how to let your agent play against any previous version of the agent (or external public agents).

#### Requirements

* [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Docker Compose](https://docs.docker.com/compose/)
* [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
* [Jupytext](https://github.com/mwouts/jupytext)

#### Getting Started

1. Copy your agents `submission_standalone.py` into the `bechmark/` directory as one of the agents:

    ```shell
    cp submission_standalone.py benchmark/agent_1_standalone.py # Example
    ```

1. Move to the `benchmark/` directory and start the containers:

    ```shell
    cd benchmark/ && docker-compose up --build
    ```

1. Move back to the project root and start Jupyter Lab.

    ```shell
    cd .. && jupyter lab
    ```

1. Open the `notebooks/benchmark.py` with Jupytext and execute it.

### Processing Time

This section demonstrates how to evaluate the processing time of your agent.

#### Requirements

* [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
* [Jupytext](https://github.com/mwouts/jupytext)

#### Getting Started

1. Make sure that the `data/test_sets` directory contain six files in the format `Test_L.*`, otherwise execute `bin/request_test_sets`.

1. Start Jupyter Lab. 
    
    ```shell
    jupyter lab
    ```

1. Open the `notebooks/processing_time_benchmark.py` with Jupytext and execute it.

## Additional Resources

* [Creating the (nearly) perfect connect-four bot with limited move time and file size](https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0)
* [Solving Connect 4: how to build a perfect AI](http://blog.gamesolver.org/)

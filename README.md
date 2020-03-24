# connect-x

My contribution to the [Connect X](https://www.kaggle.com/c/connectx) competition on [Kaggle](https://www.kaggle.com/).

## Getting Started

### Requirements

- You have stored the Kaggle API credentials file according the [official instructions](https://github.com/Kaggle/kaggle-api#api-credentials).
- You have [Poetry](https://github.com/python-poetry/poetry) installed in your global Python environment.

### Installation

1. Clone the repository from GitHub.

	```
	git clone git@github.com:janjagusch/connect-x.git
	```

1. Install the dependencies.

	```
	poetry install
	```

## Usage

1. Build your own Connect X agent.
1. Make sure your `submission.py` is valid.

	```
	poetry run python bin/validate_submission
	```

1. Submit.

	```
	poetry run bin/submit
	```

Note: You can also submit solutions through the CI when releasing a new tag. In order to do so, you new to have the `KAGGLE_API_TOKEN` variable set.

1. Create a new token for the [Kaggle API](https://github.com/Kaggle/kaggle-api) and store it as `kaggle.json`.
1. In the directory where `kaggle.json` is stored, run:

	```
	cat kaggle.json | base64
	```
1. Set the output as `KAGGLE_API_TOKEN`.

## Help and Support

### Authors

- Jan-Benedikt Jagusch <jan.jagusch@gmail.com>

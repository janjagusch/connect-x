# connect-x

My contribution to the [Connect X](https://www.kaggle.com/c/connectx) competition on [Kaggle](https://www.kaggle.com/).

## Getting Started

### Requirements

- You have stored the Kaggle API credentials file according the [official instructions](https://github.com/Kaggle/kaggle-api#api-credentials).
- You have [Poetry](https://github.com/python-poetry/poetry) installed in your global Python.

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
	poetry run bin/validate_submission
	```

1. Submit.

	```
	poetry run bin/submit
	```

## Help and Support

### Authors

- Jan-Benedikt Jagusch <jan.jagusch@gmail.com>

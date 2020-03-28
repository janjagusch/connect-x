"""
This module makes sure that the submission file is valid.
"""

import sys

from kaggle_environments import make, utils


class ValidationError(Exception):
    """
    Exception that is thrown when validaiting `submission.py` fails.
    """


def main():
    """
    Makes sure that the `submission_standalone.py` file is valid.
    """
    out = sys.stdout
    submission = utils.read_file("submission_standalone.py")
    agent = utils.get_last_callable(submission)
    sys.stdout = out

    env = make("connectx", debug=True)
    env.run([agent, agent])

    if not env.state[0].status == env.state[1].status == "DONE":
        raise ValidationError(
            "`submission_standalone.py` file is not vaild. ",
            f"agent #1 state: '{env.state[0].status}', ",
            f"agent #2 state: '{env.state[1].status}'",
        )

    print("`submission_standalone.py` file is valid.")


if __name__ == "__main__":
    main()

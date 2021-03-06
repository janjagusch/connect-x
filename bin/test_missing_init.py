"""
This module makes sure that therer are no missing `__init__.py` files in the package.
"""

import os


class MissingInitError(Exception):
    """
	A custom error class when an `__init__.py`
	file is missing.
	"""


def find_missing_init(file_path):
    """
	Finds missing `__init__.py` files in a path.

	An `__init__.py` is missing, when there are other `*.py`
	files in the directory.

	Args:
		file_path (str): a file path.

	Returns:
		list: a list of file paths where the `__init__.py`
			is missing.
	"""
    missing = []
    files = os.listdir(file_path)
    if any(file.endswith(".py") for file in files):
        if "__init__.py" not in files:
            missing.append(file_path)
        for file in files:
            new_file_path = os.path.join(file_path, file)
            if os.path.isdir(new_file_path):
                missing.extend(find_missing_init(new_file_path))
    return missing


if __name__ == "__main__":
    MISSING_INIT = find_missing_init("connect_x")
    if len(MISSING_INIT) > 0:
        raise MissingInitError("__init__.py is missing in: {}".format(MISSING_INIT))

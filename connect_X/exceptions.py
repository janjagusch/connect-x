"""
This module contains costum exceptions for the `connect-x` project.
"""

class ConnectXException(Exception):
    """
    Base exception for the `connect-x` project.
    """


class ValidationError(ConnectXException):
    """
    Exception when validating `submission.py` fails.
    """


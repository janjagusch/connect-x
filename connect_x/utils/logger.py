"""
Provides a utils method to setup the logger.
"""
import os
import logging


def setup_logger(logger_name=None):
    """
    Get the logger.

    Args:
        logger_name (str): The name of the logger.
    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(name)s - %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M%S %p",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger

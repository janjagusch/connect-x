"""
This module contains pre-calculated board actions.
"""
import gzip
import json

FORECAST_DEPTH = 5

# pylint: disable=line-too-long
_BOARD_ACTION_MAP_BINARY = b"\x1f\x8b\x08\x00\xc7\xad\x7f^\x02\xff\xed\xd7\xb1\r\x800\x10\x04\xc1V,\xc7\x04\x08\x88\\\x0be@\x84\xe8\x9d\xa7\x0bt\x8c\x83)\xc0\xc9\xde_}?\xe7z$I2\xc9>\xda:5\x99'I~\xc3\xc3\x1fH<I\x92&\x81\xc4\x93$\xf9\xb7IP\x89_$\x9e$\xc9\xb8I\xe0\x8a'I2r\x12T\xe27\x89'I2n\x12\xb8\xe2I\x92\x8c\x9c\x04o\xe2\xef\x07\x83\xc4(y\x1c?\x00\x00"

# pylint: enable=line-too-long
BOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))

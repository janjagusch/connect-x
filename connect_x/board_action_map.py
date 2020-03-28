"""
This module contains pre-calculated board actions.
"""
import gzip
import json

FORECAST_DEPTH = 5

# pylint: disable=line-too-long
_BOARD_ACTION_MAP_BINARY = b'\x1f\x8b\x08\x006q\x7f^\x02\xff\xabV2 \x1a(Y)\x18\xeb(\x10\xa3\xc1\x92T\rP\x1d$h\x80\xe8\x00j0"V\x03X\x07)6\x80u\x005\x98\x10\xaf\x01\xa8\x834\x1b\x80:@\x1aj\x01\xbduq|\x88\x01\x00\x00'

# pylint: enable=line-too-long
BOARD_ACTION_MAP = json.loads(gzip.decompress(_BOARD_ACTION_MAP_BINARY))

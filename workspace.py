from os import path
from typing import *
from itertools import count

from shared import *

EXP_PATH = path.join('./experiments/',
'''
2023_m05_d17@08_36_31_nitf_sees_not
'''
.strip())

TAKE_EVERY = 128

# TIME_SLICE = slice(None)
TIME_SLICE = slice(0, 256)
# EPOCHS = count()
# EPOCHS = range(100, 110)
EPOCHS = count(((25000 // TAKE_EVERY) * TAKE_EVERY), 256)
VOICED_PAGE_I = 8000 // PAGE_LEN
SELECT_GROUPS = slice(None)
# SELECT_GROUPS = slice(-1, None)

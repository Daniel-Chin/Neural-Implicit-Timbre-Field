from os import path
from typing import *
from itertools import count

from shared import *

EXP_PATH = path.join('./experiments/',
'''
2023_m05_d16@07_38_43_dredge_test
'''
.strip())

# TIME_SLICE = slice(None)
TIME_SLICE = slice(0, 256)
# EPOCHS = count()
# EPOCHS = range(100, 110)
EPOCHS = count(250, 10)
VOICED_PAGE_I = 8000 // PAGE_LEN

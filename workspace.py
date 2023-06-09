from os import path
from typing import *
from itertools import count

from shared import *

EXP_PATH = path.join('./experiments/',
'''
2023_m05_d29@01_08_19_mss
'''
.strip())

def EPOCHS(experiment):
    # start = 0
    start = 300
    # step = experiment.SLOW_EVAL_EPOCH_INTERVAL
    step = 8
    real_start = round(start / step) * step
    return count(real_start, step)
    # return range(real_start, real_start + step * 8, step)
    # return range(real_start, 3000, step)

# TIME_SLICE = slice(None)
TIME_SLICE = slice(0, 256)
SELECT_PAGE = round(.25 * SR / PAGE_LEN * 2) # scale
# SELECT_PAGE = round(1.35 * SR / PAGE_LEN * 2) # scale
# SELECT_PAGE = round(5.3 * SR / PAGE_LEN * 2) # dan
# SELECT_PAGE = round(19.5 * SR / PAGE_LEN * 2) # jupyter
SELECT_GROUPS = slice(None)
# SELECT_GROUPS = slice(1, None)

SELECT_NITF = 0

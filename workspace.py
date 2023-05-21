from os import path
from typing import *
from itertools import count

from shared import *

EXP_PATH = path.join('./experiments/',
'''
2023_m05_d19@21_21_44_not_latent
'''
.strip())

def EPOCHS(experiment):
    # start = 0
    start = 80
    step = experiment.SLOW_EVAL_EPOCH_INTERVAL
    # step = 128
    real_start = round(start / step) * step
    return count(real_start, step)
    # return range(real_start, real_start + step * 4, step)

# TIME_SLICE = slice(None)
TIME_SLICE = slice(0, 256)
SELECT_PAGE = round(.25 * SR / PAGE_LEN * 2)
# SELECT_PAGE = round(1.35 * SR / PAGE_LEN * 2)
SELECT_GROUPS = slice(None)
# SELECT_GROUPS = slice(-1, None)

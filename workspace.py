from os import path
from typing import *
from itertools import count

EXP_PATH = path.join('./experiments/',
'''
2023_m05_d15@08_22_26_f0_is_latent
'''
.strip())

# TIME_SLICE = slice(None)
TIME_SLICE = slice(0, 256)
# EPOCHS = count()
EPOCHS = range(100, 110)

from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree

from prepare import *
from hyper_params import *
from exp_group import ExperimentGroup

SLOW_EVAL_EPOCH_INTERVAL = 1

EXP_NAME = 'nif_width'
N_RAND_INITS = 2
SONG_FILENAME = 'yanhe.wav'
f0s, timbres, amps, dataset = prepare(SONG_FILENAME)
N_PAGES = len(f0s)

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'nif_width'
        self.variable_value = hyperParams.nif_width
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('harmonics', 1, None), 
])
template.lr = 1e-2
template.weight_decay = 1e-9
template.optim_name = 'adam'
template.nif_width = 128
template.nif_depth = 6
template.n_vowel_dims = 2
template.batch_size = 2 ** 12
template.max_epoch = 10000

for nif_width in [64, 128, 256]:
    hP = deepcopy(template)
    hP.nif_width = nif_width
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))

from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree

from shared import *
from hyper_params import *
from exp_group import ExperimentGroup
from dataset import MyDataset

from dataset_definitions import danF0NotLatent as datasetDef
SLOW_EVAL_EPOCH_INTERVAL = 1

EXP_NAME = ...
N_RAND_INITS = 8
dataset = MyDataset(datasetDef)

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = ...
        self.variable_value = hyperParams.???
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('harmonics', 1, None), 
    LossWeightTree('dredge_regularize', 0, None), 
])
template.lr = 1e-3
template.weight_decay = 1e-9
template.optim_name = 'adam'
template.nif_width = 128
template.nif_depth = 6
template.n_vowel_dims = 2
template.nif_sees_f0 = True
template.nif_sees_amp = True
template.nif_sees_vowel = True
template.nif_abs_out = False
template.nif_abs_confidence = False
template.nif_renorm_confidence = True
template.nif_fast_lr = 2e-2
template.ground_truth_f0 = None
template.batch_size = 2 ** 12
template.max_epoch = 10000

if DEBUG_CUT_CORNERS:
    template.batch_size //= 8

for xxx in []:
    hP = deepcopy(template)
    hP.xxx = xxx
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))

from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree

from shared import *
from hyper_params import *
from exp_group import ExperimentGroup
from dataset import MyDataset

from dataset_definitions import voiceScaleF0IsLatent as datasetDef
SLOW_EVAL_EPOCH_INTERVAL = 128

EXP_NAME = 'ground_truth_f0'
N_RAND_INITS = 2
dataset = MyDataset(datasetDef)

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'nif_abs_out'
        self.variable_value = hyperParams.nif_abs_out
    
    @lru_cache(1)
    def name(self):
        return f'{self.variable_name}={self.variable_value}'

GROUPS = []

template = HyperParams()
template.lossWeightTree = LossWeightTree('total', 1, [
    LossWeightTree('harmonics', 1, None), 
    LossWeightTree('dredge_regularize', 1e-6, None), 
])
template.lr = 1e-3
template.weight_decay = 1e-9
template.optim_name = 'adam'
template.nif_width = 128
template.nif_depth = 6
template.n_vowel_dims = 2
template.nif_sees_f0 = False
template.nif_sees_amp = False
template.nif_sees_vowel = False
template.nif_abs_out = False
template.ground_truth_f0 = False
template.batch_size = 256
template.max_epoch = 1e5

template.ground_truth_f0 = True

for nif_abs_out in [False, True]:
    hP = deepcopy(template)
    hP.nif_abs_out = nif_abs_out
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))

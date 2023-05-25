from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree

from shared import *
from hyper_params import *
from exp_group import ExperimentGroup
from dataset import MyDataset

from dataset_definitions import voiceScaleF0IsLatent as datasetDef
SLOW_EVAL_EPOCH_INTERVAL = 1

EXP_NAME = 'fast_lr'
N_RAND_INITS = 1
dataset = MyDataset(datasetDef)

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'nif_fast_lr'
        self.variable_value = (
            hyperParams.nif_fast_lr, 
        )
    
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
template.nif_abs_confidence = False
template.nif_renorm_confidence = True
template.nif_fast_lr = 2e-2
template.ground_truth_f0 = False
template.batch_size = 256
template.max_epoch = 1e5

template.lossWeightTree['dredge_regularize'].weight = 1e-5
template.nif_abs_out = True
template.nif_abs_confidence = True

for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
    hP = deepcopy(template)
    template.nif_fast_lr = lr
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))

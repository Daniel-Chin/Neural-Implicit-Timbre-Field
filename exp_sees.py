from typing import *
from functools import lru_cache
from copy import deepcopy

from torchWork import LossWeightTree

from shared import *
from hyper_params import *
from exp_group import ExperimentGroup
from dataset import MyDataset

from dataset_definitions import danF0IsLatent as datasetDef
SLOW_EVAL_EPOCH_INTERVAL = 1
LOG_SAMPLE_PAGE = False

EXP_NAME = 'sees'
N_RAND_INITS = 3
dataset = MyDataset(datasetDef)

class MyExpGroup(ExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams

        self.variable_name = 'nif_sees_f0'
        self.variable_value = hyperParams.nif_sees_f0
    
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
template.n_nifs = 1
template.nif_width = 128
template.nif_depth = 6
template.n_vowel_dims = 2
template.nif_sees_f0 = True
template.nif_sees_amp = True
template.nif_sees_vowel = True
template.nif_abs_out = True
template.nif_abs_confidence = False
template.nif_renorm_confidence = True
template.latent_low_lr = 1e-3
template.latent_high_lr = 1e-2
template.ground_truth_f0 = False
template.batch_size = 256
template.max_epoch = 1e3

for nif_sees_f0 in [False, True]:
    hP = deepcopy(template)
    hP.nif_sees_f0 = nif_sees_f0
    hP.ready(globals())
    GROUPS.append(MyExpGroup(hP))

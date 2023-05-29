from typing import *

from torchWork.experiment_control import loadLatestModels

from nitf import NITF
from dataset_definitions import DatasetDefinition
from exp_group import ExperimentGroup

def loadNITFForEval(
        exp_path, datasetDef: DatasetDefinition, 
        group: ExperimentGroup, rand_init_i, epoch, 
    ):
    hParams = group.hyperParams
    epoch, models = loadLatestModels(
        exp_path, group, rand_init_i, dict(
            nitf=(NITF, hParams.n_nifs), 
        ), epoch, verbose = epoch is None, 
    )
    nitfs: List[NITF] = models['nitf']
    for nitf in nitfs:
        nitf.eval()
        nitf.vowel_embs = nitf.get_buffer('saved_vowel_embs')
        if datasetDef.is_f0_latent:
            nitf.amp_latent        = nitf.get_buffer('saved_amp_latent')
            if not nitf.hParams.ground_truth_f0:
                nitf.dredge_freq       = nitf.get_buffer('saved_dredge_freq')
                nitf.dredge_confidence = nitf.get_buffer('saved_dredge_confidence')
    return nitfs

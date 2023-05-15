from torchWork.experiment_control import loadLatestModels

from nitf import NITF
from dataset_definitions import DatasetDefinition

def loadNITFForEval(
        exp_path, datasetDef: DatasetDefinition, 
        group, rand_init_i, epoch, 
    ):
    epoch, models = loadLatestModels(
        exp_path, group, rand_init_i, dict(
            nitf=(NITF, 1), 
        ), epoch, verbose = epoch is None, 
    )
    nitf: NITF = models['nitf'][0]
    nitf.eval()
    nitf.vowel_embs = nitf.get_buffer('saved_vowel_embs')
    if datasetDef.is_f0_latent:
        nitf.f0_latent  = nitf.get_buffer('saved_f0_latent')
        nitf.amp_latent = nitf.get_buffer('saved_amp_latent')
    return nitf

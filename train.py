from __future__ import annotations

from typing import *
from functools import lru_cache

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, Optimizer
import torchWork
from torchWork import runExperiment, Profiler, LossLogger, saveModels, HAS_CUDA, loadExperiment, DEVICE

from shared import *
from losses import Loss_root
from hyper_params import HyperParams
from arg_parser import ArgParser
from nitf import NITF
from dataset_definitions import DatasetDefinition
from dataset import MyDataset
from exp_group import ExperimentGroup
from lobe import getLobe
from dredge import *
from log_sample_page import logSamplePage

def requireModelClasses(hParams: HyperParams):
    x = {}
    x['nitf'] = (NITF, hParams.n_nifs)
    return x

def main():
    args = ArgParser()
    print(f'{args.exp_py_path = }')
    exp_py_path = args.exp_py_path

    (
        experiment_name, n_rand_inits, groups, experiment, 
    ) = loadExperiment(exp_py_path)
    print(
        'Experiment:', experiment_name, ',', 
        len(groups), 'x', n_rand_inits, 
    )

    dataset: MyDataset = experiment.dataset
    group: ExperimentGroup = groups[0]
    hParams: HyperParams = group.hyperParams
    print(
        f'In group {group.name()}, # of batches per epoch =', 
        len(dataset) / hParams.batch_size, 
    )

    runExperiment(
        exp_py_path, requireModelClasses, oneEpoch, 
        dataset, None, 
    )

def oneEpoch(
    group_name: str, epoch: int, 
    experiment, hParams: HyperParams, 
    models: Dict[str, List[torch.nn.Module]], 
    optim: torch.optim.Optimizer, 
    trainSet: MyDataset, _: Dataset, 
    lossLogger: LossLogger, profiler: Profiler, 
    save_path: str, trainer_id: int, 
):
    datasetDef: DatasetDefinition = experiment.datasetDef
    nitfs: List[NITF] = models['nitf']
    [x.train() for x in nitfs]
    dataLoader = DataLoader(
        trainSet, hParams.batch_size, shuffle=True, 
        drop_last=True, 
    )
    oneshot_optims: List[Optimizer] = []
    for nitf in nitfs:
        slowOptim = Adam(
            nitf.low_lr_latents, hParams.latent_low_lr, 
        )
        oneshot_optims.append(slowOptim)
        fastOptim = Adam(
            nitf.high_lr_latents, hParams.latent_high_lr, 
        )
        oneshot_optims.append(fastOptim)
    for batch_i, batch in enumerate(dataLoader):
        if DEBUG_CUT_CORNERS and batch_i == 4:
            print('DEBUG_CUT_CORNERS')
            break
        lossTree = Loss_root()
        if datasetDef.is_f0_latent:
            batchF0IsLatent(lossTree, nitfs, trainSet, hParams, *batch)
        else:
            assert hParams.n_nifs == 1
            batchF0NotLatent(lossTree, hParams, nitfs[0], *batch)
        optim.zero_grad()
        [x.zero_grad() for x in oneshot_optims]
        total_loss = lossTree.sum(hParams.lossWeightTree, epoch)
        total_loss.backward()
        optim.step()
        [x.step() for x in oneshot_optims]
        if hParams.nif_renorm_confidence:
            [x.renormConfidence() for x in nitfs]

        lossLogger.eat(
            epoch, batch_i, True, profiler, lossTree, 
            hParams.lossWeightTree, 
        )

    [x.eval() for x in nitfs]
    with torch.no_grad():
        if trainSet.datasetDef.is_f0_latent and epoch % 16 == 0:
            [x.simplifyDredge(fastOptim) for x in nitfs]
        
        if epoch % experiment.SLOW_EVAL_EPOCH_INTERVAL == 0:
            saveModels(models, epoch, save_path)
        if epoch < 4 or (epoch ** .5).is_integer():
            print(group_name, 'epoch', epoch, 'finished.', flush=True)
            # print('last batch loss =', total_loss.item(), flush=True)
        
        if experiment.LOG_SAMPLE_PAGE and datasetDef.is_f0_latent:
            logSamplePage(
                epoch, nitfs, hParams, trainSet, datasetDef, save_path, 
                forwardF0IsLatent, 
            )
    
    return epoch <= hParams.max_epoch

def batchF0NotLatent(
    lossTree: Loss_root, hParams: HyperParams, 
    nitf: NITF, x, y, page_i, 
):
    assert hParams.nif_sees_f0
    assert hParams.nif_sees_amp
    assert hParams.nif_sees_vowel
    nitf_in = torch.concat((
        x, nitf.vowel_embs[page_i, :], 
    ), dim=1)
    y_hat = nitf.forward(nitf_in)[:, 0]
    lossTree.harmonics = F.mse_loss(y_hat * x[:, 2], y).cpu()

@lru_cache()
def getFreqCube(batch_size, n_freq_bins):
    x = torch.arange(0, n_freq_bins).float()
    x = x.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(
        batch_size, DREDGE_LEN, N_HARMONICS, 1, 
    )
    return x.to(DEVICE)

LADDER = torch.arange(0, N_HARMONICS).float().to(
    DEVICE
).unsqueeze(0).unsqueeze(1).contiguous() + 1
def forwardF0IsLatent(
    nitf: NITF, dataset: MyDataset, hParams: HyperParams, 
    page_i, batch_size_override=None, 
):
    batch_size = batch_size_override or hParams.batch_size
    dredge_freq = freqDenorm(nitf.dredge_freq[page_i])
    dredge_confidence = nitf.dredge_confidence[page_i, :]
    if hParams.nif_abs_confidence:
        dredge_confidence = dredge_confidence.abs()
    amp = nitf.amp_latent[page_i]
    ve  = nitf.vowel_embs[page_i, :]

    f0 = dredge_freq.unsqueeze(1) * DREDGE_MULT.unsqueeze(0)
    # f0 is (batch_size, DREDGE_LEN)
    f0 = f0.unsqueeze(2).repeat(1, 1, N_HARMONICS)
    # f0 is (batch_size, DREDGE_LEN, N_HARMONICS)
    freq = f0 * LADDER
    # freq is (batch_size, DREDGE_LEN, N_HARMONICS)
    amp = amp.unsqueeze(1).unsqueeze(2).repeat(1, DREDGE_LEN, N_HARMONICS)
    # amp is (batch_size, DREDGE_LEN, N_HARMONICS)
    ve = ve.unsqueeze(1).unsqueeze(2).repeat(1, DREDGE_LEN, N_HARMONICS, 1)
    # ve is (batch_size, DREDGE_LEN, N_HARMONICS, n_vowel_dims)
    dredge_confidence = dredge_confidence.unsqueeze(2).repeat(1, 1, N_HARMONICS)
    # dredge_confidence is (batch_size, DREDGE_LEN, N_HARMONICS)
    nitf_in = [        freqNorm(freq).unsqueeze(3)]
    if hParams.nif_sees_f0:
        nitf_in.append(freqNorm(f0  ).unsqueeze(3))
    if hParams.nif_sees_amp:
        nitf_in.append(amp .unsqueeze(3))
    if hParams.nif_sees_vowel:
        nitf_in.append(ve)
    nitf_in = torch.concat(nitf_in, dim=3)
    mag = nitf.forward(nitf_in)[:, :, :, 0]
    freqCube = getFreqCube(
        batch_size, dataset.n_freq_bins, 
    )
    # freqCube is (batch_size, DREDGE_LEN, N_HARMONICS, n_freq_bins)
    freqCube = freqCube - (freq * dataset.one_over_freq_bin).unsqueeze(3)
    freqCube = getLobe()(freqCube)
    freqCube = freqCube * (
        mag * amp * dredge_confidence
    ).unsqueeze(3)
    return freqCube.sum(dim=2).sum(dim=1)

def batchF0IsLatent(
    lossTree: Loss_root, 
    nitfs: List[NITF], dataset: MyDataset, hParams: HyperParams, 
    x, page_i, 
):
    x_hats = []
    reg_losses = []
    for nitf in nitfs:
        x_hats.append(forwardF0IsLatent(
            nitf, dataset, hParams, page_i, 
        ))
        reg_losses.append(regularizeDredge(
            nitf.dredge_confidence[page_i, :], 
        ).cpu())
    x_hat = torch.stack(x_hats, dim=0).sum(dim=0)
    lossTree.harmonics = F.mse_loss(x_hat, x).cpu()
    lossTree.dredge_regularize = torch.stack(reg_losses, dim=0).mean(dim=0)

if __name__ == '__main__':
    main()

from __future__ import annotations

from typing import *
from functools import lru_cache

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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

def requireModelClasses(_):
    x = {}
    x['nitf'] = (NITF, 1)
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
    nitf: NITF = models['nitf'][0]
    nitf.train()
    dataLoader = DataLoader(
        trainSet, hParams.batch_size, shuffle=True, 
        drop_last=True, 
    )
    for batch_i, batch in enumerate(dataLoader):
        if datasetDef.is_f0_latent:
            losses = batchF0IsLatent(nitf, trainSet, hParams, *batch)
        else:
            losses = batchF0NotLatent(nitf, *batch)
        optim.zero_grad()
        for loss in losses:
            loss.backward()
        optim.step()

        lossTree = Loss_root()
        lossTree.harmonics = loss.cpu()
        lossLogger.eat(
            epoch, batch_i, True, profiler, lossTree, 
            hParams.lossWeightTree, 
        )

    saveModels(models, epoch, save_path)
    if epoch < 4 or (epoch ** .5).is_integer():
        print(group_name, 'epoch', epoch, 'finished.')
        print('last batch loss =', loss.item(), flush=True)
    
    return True

def batchF0NotLatent(nitf: NITF, x, y, page_i):
    nitf_in = torch.concat((
        x, nitf.vowel_embs[page_i, :], 
    ), dim=1)
    y_hat = nitf.forward(nitf_in)
    yield F.mse_loss(y_hat[:, 0], y)

@lru_cache()
def getFreqCube(batch_size, n_freq_bins):
    x = torch.arange(0, n_freq_bins).float()
    x = x.unsqueeze(0).repeat(N_HARMONICS, 1)
    x = x.unsqueeze(0).repeat(batch_size, 1, 1)
    return x.to(DEVICE)

LADDER = torch.arange(0, N_HARMONICS).float().unsqueeze(
    0
).to(DEVICE).contiguous() + 1
def forwardF0IsLatent(
    nitf: NITF, dataset: MyDataset, hParams: HyperParams, 
    page_i, batch_size_override=None, 
):
    batch_size = batch_size_override or hParams.batch_size
    f0  = nitf. f0_latent[page_i]
    amp = nitf.amp_latent[page_i]
    ve  = nitf.vowel_embs[page_i, :]

    freq = f0.unsqueeze(1) @ LADDER
    # freq is (batch_size, N_HARMONICS)
    nitf_in = torch.concat((
        freq.unsqueeze(2), 
        f0 .unsqueeze(1).repeat(1, N_HARMONICS).unsqueeze(2), 
        amp.unsqueeze(1).repeat(1, N_HARMONICS).unsqueeze(2), 
        ve .unsqueeze(1).repeat(1, N_HARMONICS, 1), 
    ), dim=2)
    mag = nitf.forward(nitf_in)[:, :, 0]
    freqCube = getFreqCube(
        batch_size, dataset.n_freq_bins, 
    )
    freqCube = freqCube - (freq * dataset.one_over_freq_bin).unsqueeze(2)
    freqCube = getLobe()(freqCube)
    freqCube = freqCube * (mag * amp.unsqueeze(1)).unsqueeze(2)
    return freqCube.sum(dim=1)

def batchF0IsLatent(
    nitf: NITF, dataset: MyDataset, hParams: HyperParams, 
    x, page_i, 
):
    x_hat = forwardF0IsLatent(nitf, dataset, hParams, page_i)
    yield F.mse_loss(x_hat, x)

if __name__ == '__main__':
    main()

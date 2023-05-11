from __future__ import annotations

from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchWork
from torchWork import runExperiment, Profiler, LossLogger, saveModels, HAS_CUDA, loadExperiment

from prepare import *
from losses import Loss_root
from hyper_params import HyperParams
from arg_parser import ArgParser

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

    runExperiment(
        exp_py_path, requireModelClasses, oneEpoch, 
        experiment.dataset, None, 
    )

def oneEpoch(
    group_name: str, epoch: int, 
    experiment, hParams: HyperParams, 
    models: Dict[str, List[torch.nn.Module]], 
    optim: torch.optim.Optimizer, 
    trainSet: Dataset, validateSet: Dataset, 
    lossLogger: LossLogger, profiler: Profiler, 
    save_path: str, trainer_id: int, 
):
    nitf: NITF = models['nitf'][0]
    nitf.train()
    dataLoader = DataLoader(trainSet, hParams.batch_size, shuffle=True)
    for batch_i, (x, y, page_i) in enumerate(dataLoader):
        x_vowel = torch.concat((
            x, nitf.vowel_embs[page_i], 
        ), dim=1)
        y_hat = nitf.forward(x_vowel)
        loss = F.mse_loss(y_hat[:, 0], y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        lossTree = Loss_root()
        lossTree.harmonics = loss.cpu()
        lossLogger.eat(
            epoch, batch_i, True, profiler, lossTree, 
            hParams.lossWeightTree, 
        )

    saveModels(models, epoch, save_path)
    if epoch % 8 == 0:
        print(group_name, 'epoch', epoch, 'finished.')
    
    return True

if __name__ == '__main__':
    main()

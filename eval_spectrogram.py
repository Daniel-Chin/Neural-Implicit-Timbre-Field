from os import path
from typing import *

import numpy as np
import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, getTrainerPath
from matplotlib import pyplot as plt
from tqdm import tqdm

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval
from train import forwardF0IsLatent

from workspace import EXP_PATH, TIME_SLICE, EPOCHS, SELECT_GROUPS

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        dataset: MyDataset = experiment.dataset
        
        for group in tqdm(groups[SELECT_GROUPS]):
            print(group.name())
            for rand_init_i in range(n_rand_inits):
            # rand_init_i = 0
            # if True:
                for epoch in EPOCHS(experiment):
                    try:
                        nitfs = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError:
                        break
                    print(f'{epoch = }')
                    evalOne(nitfs, dataset, group.hyperParams)
                    # plt.show()
                    plt.savefig(path.join(
                        getTrainerPath(
                            EXP_PATH, 
                            group.pathName(), 
                            rand_init_i, 
                        ), 
                        f'spectrogram_{epoch}.png', 
                    ), dpi=200)

def evalOne(nitfs: List[NITF], dataset: MyDataset, hParams):
    fig = plt.figure()
    axes = fig.subplots(2, 1, sharex=True)
    axes: List[plt.Axes] = axes.tolist()
    times = dataset.times[TIME_SLICE]
    data = dataset.X.T[:, TIME_SLICE]
    # data = data.log()
    vmin = data.min()
    vmax = data.max()
    # print('plot 0...')
    axes[0].pcolormesh(
        times, dataset.freqs, data, 
        vmin=vmin, vmax=vmax, 
    )
    axes[0].set_title('Ground truth')
    axes[0].set_xlabel('Time (sec)')
    axes[0].set_ylabel('Freq (Hz)')
    # print('forward...')
    x_hats = []
    for nitf in nitfs:
        x_hats.append(forwardF0IsLatent(
            nitf, dataset, hParams, dataset.I[TIME_SLICE], 
            batch_size_override=len(times),
        ))
    x_hat = torch.stack(x_hats, dim=0).sum(dim=0)
    # print('plot 1...')
    data_1 = x_hat.T.clip(1e-8)
    # data_1 = data_1.log()
    pcm = axes[1].pcolormesh(
        times, dataset.freqs, data_1, 
        vmin=vmin, vmax=vmax, 
    )
    axes[1].set_title('Recon')
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Freq (Hz)')
    fig.tight_layout()
    fig.colorbar(pcm, ax=axes)

if __name__ == '__main__':
    main()

from os import path
from typing import *
from itertools import count

import numpy as np
import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
from matplotlib import pyplot as plt

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval
from train import forwardF0IsLatent

from workspace import EXP_PATH, TIME_SLICE

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        dataset: MyDataset = experiment.dataset
        
        for group in groups:
            print(group.name())
            # for rand_init_i in range(n_rand_inits):
            rand_init_i = 0
            if True:
                for epoch in count():
                    try:
                        nitf = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError:
                        break
                    print(f'{epoch = }')
                    evalOne(nitf, dataset, group.hyperParams)

def evalOne(nitf: NITF, dataset: MyDataset, hParams):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes: List[plt.Axes] = axes.tolist()
    times = dataset.times[TIME_SLICE]
    data = dataset.X.T[:, TIME_SLICE]
    vmax = data.max()
    print('plot 0...')
    axes[0].pcolormesh(
        times, dataset.freqs, data, 
        vmin=0, vmax=vmax, 
    )
    axes[0].set_title('Ground truth')
    axes[0].set_xlabel('Time (sec)')
    axes[0].set_ylabel('Freq (Hz)')
    print('forward...')
    x_hat = forwardF0IsLatent(
        nitf, dataset, hParams, dataset.I[TIME_SLICE], 
        batch_size_override=len(times),
    )
    print('plot 1...')
    pcm = axes[1].pcolormesh(
        times, dataset.freqs, x_hat.T, 
        vmin=0, vmax=vmax, 
    )
    axes[1].set_title('Recon')
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Freq (Hz)')
    fig.colorbar(pcm, ax=axes)
    plt.show()

if __name__ == '__main__':
    main()

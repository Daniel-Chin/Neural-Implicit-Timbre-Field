from os import path
from typing import *
from itertools import count

import numpy as np
import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
from matplotlib import pyplot as plt

from import_dan_py import ImportDanPy
with ImportDanPy():
    pass

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval

from workspace import (
    EXP_PATH, EPOCHS, SELECT_GROUPS, TIME_SLICE, 
)

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        dataset: MyDataset = experiment.dataset
        truth_t = dataset.times
        truth_f0 = dataset.f0_tracks
        truth_amp = []
        for t in truth_t:
            start = round(SR * t.item())
            page = dataset.wav[start : start + PAGE_LEN]
            if len(page) == 0:
                # stft zero pad
                amp = 0
            else:
                amp = np.sqrt(np.square(page).mean())
            truth_amp.append(amp)
        # C = colorLadder(len(groups))
        C = colorLadder(2)
        
        def plotTruth():
            kw = dict(label='Ground truth')
            for track_i, t_f0 in enumerate(truth_f0):
                plt.plot(
                    truth_t, t_f0, 
                    'ox'[track_i], 
                    markersize=8, 
                    markerfacecolor='none', markeredgecolor='k', 
                    markeredgewidth=.5, 
                    **kw, 
                )
                kw.clear()
            # plt.plot(
            #     truth_t, truth_amp, 
            #     'o', linewidth=.5, markersize=.5, 
            #     label='Ground truth', 
            # )
        
        for epoch in EPOCHS(experiment):
            print(f'{epoch = }')
            # plotTruth()
            for group_i, group in enumerate(groups[SELECT_GROUPS]):
                # kw = dict(label=group.name())
                print(group.name())
                for rand_init_i in range(n_rand_inits):
                # rand_init_i = 0
                # if True:
                    print(f'{rand_init_i = }')
                    try:
                        nitfs = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError:
                        return
                    plotTruth()
                    for nitf_i, nitf in enumerate(nitfs):
                        plt.plot(
                            dataset.times, 
                            freqDenorm(nitf.dredge_freq), 
                            # nitf.amp_latent, 
                            # 'xv^s.'[group_i], 
                            '^v'[nitf_i], 
                            markersize=6, markerfacecolor='none', 
                            markeredgewidth=.5, 
                            # c=C[group_i], 
                            c=C[nitf_i], 
                            # **kw, 
                        )
                    # kw.clear()
                    plt.legend()
                    plt.show()
            # plt.yscale('log')
            # plt.legend()
            # plt.show()

if __name__ == '__main__':
    main()

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
    from yin import yin

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval

from workspace import EXP_PATH, EPOCHS, SELECT_GROUPS, TIME_SLICE

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        dataset: MyDataset = experiment.dataset
        truth_t = []
        truth_f0 = []
        truth_amp = []
        for i, page in enumerate(pagesOf(dataset.wav)):
            f0 = yin(page, SR, PAGE_LEN, fmin=80, fmax=800)
            amp = np.sqrt(np.square(page).sum())
            truth_t.append(i * PAGE_LEN / SR)
            truth_f0.append(f0)
            truth_amp.append(amp)
        
    def f():
        for epoch in EPOCHS:
            print(f'{epoch = }')
            plt.plot(
                truth_t, truth_f0, 
                'o', 
                markersize=8, markerfacecolor='none', 
                markeredgewidth=.5, 
                label='YIN', 
            )
            # plt.plot(
            #     truth_t, truth_amp, 
            #     'o', linewidth=.5, markersize=.5, 
            #     label='Ground truth', 
            # )
            for group_i, group in enumerate(groups[SELECT_GROUPS]):
                kw = dict(label=group.name())
                # for rand_init_i in range(n_rand_inits):
                rand_init_i = 0
                if True:
                    try:
                        nitf = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError:
                        return
                    plt.plot(
                        dataset.times, 
                        nitf.dredge_freq * FREQ_SCALE, 
                        # nitf.amp_latent, 
                        'xv^s.'[group_i], 
                        markersize=6, markerfacecolor='none', 
                        markeredgewidth=.5, 
                        **kw, 
                    )
                    kw.clear()
            # plt.yscale('log')
            plt.legend()
            plt.show()
        
    with torch.no_grad():
        f()

if __name__ == '__main__':
    main()

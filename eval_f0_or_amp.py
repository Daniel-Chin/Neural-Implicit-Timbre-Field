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

from workspace import EXP_PATH

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
            f0 = yin(page, SR, PAGE_LEN)
            amp = np.sqrt(np.square(page).sum())
            truth_t.append(i * PAGE_LEN / SR)
            truth_f0.append(f0)
            truth_amp.append(amp)
        
    def f():
        for epoch in count(step=10):
            print(f'{epoch = }')
            # plt.plot(truth_t, truth_f0, linewidth=.5, label='YIN')
            # plt.plot(truth_t, truth_amp, linewidth=.5, label='Ground truth')
            for group in groups:
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
                        nitf.dredge_freq, 
                        # nitf.amp_latent, 
                        linewidth=.5, 
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

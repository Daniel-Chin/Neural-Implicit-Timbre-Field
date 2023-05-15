from os import path
from typing import *

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
        yin_t = []
        yin_f0 = []
        for i, page in enumerate(pagesOf(dataset.wav)):
            f0 = yin(page, SR, PAGE_LEN)
            yin_t.append(i * PAGE_LEN / SR)
            yin_f0.append(f0)
        plt.plot(yin_t, yin_f0, linewidth=.5, label='YIN')
        
        for group in groups:
            kw = dict(label=group.name())
            # for rand_init_i in range(n_rand_inits):
            rand_init_i = 0
            if True:
                nitf = loadNITFForEval(
                    EXP_PATH, experiment.datasetDef, 
                    group, rand_init_i, None, 
                )
                plt.plot(
                    dataset.times, 
                    nitf.f0_latent, 
                    linewidth=.5, 
                    **kw, 
                )
                kw.clear()
        
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()

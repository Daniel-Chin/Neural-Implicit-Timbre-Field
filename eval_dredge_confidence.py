from os import path
from typing import *
from itertools import count

import numpy as np
import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
from matplotlib import pyplot as plt
from tqdm import tqdm

from import_dan_py import ImportDanPy
with ImportDanPy():
    from yin import yin

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval
from dredge import *

from workspace import EXP_PATH, EPOCHS, SELECT_PAGE

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
                print(f'{rand_init_i = }')
                X = []
                final_f0 = None
                for epoch in tqdm(EPOCHS(experiment)):
                    try:
                        nitf = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError:
                        break
                    X.append(nitf.dredge_confidence[SELECT_PAGE, :])
                    final_f0 = freqDenorm(nitf.dredge_freq[SELECT_PAGE].item())
                print(f'{final_f0 = }')
                X = torch.stack(X, dim=0).T
                for i in range(DREDGE_LEN):
                    plt.plot(
                        X[i, :], 
                        'o', linewidth=.5, markersize=.5, 
                        label=f'{DREDGE_MULT[i]:.2f}', 
                    )
                plt.legend()
                plt.show()

if __name__ == '__main__':
    main()

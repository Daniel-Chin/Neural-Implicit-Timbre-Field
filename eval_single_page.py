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

R = np.linspace(0, 1, DREDGE_LEN)
G = np.linspace(1, 0, DREDGE_LEN)
B = np.linspace(0, 0, DREDGE_LEN)
C = [*zip(R, G, B)]
C[DREDGE_RADIUS] = 'k'

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        assert next(EPOCHS(experiment)) == 0
        dataset: MyDataset = experiment.dataset
        
        audio= dataset.wav[
            PAGE_LEN * SELECT_PAGE : 
            PAGE_LEN * (SELECT_PAGE + 1)
        ]
        yin_f0 = yin(audio, SR, PAGE_LEN)
        print(f'{yin_f0 = }')
        
        for group in groups:
            print(group.name())
            # for rand_init_i in range(n_rand_inits):
            rand_init_i = 0
            if True:
                print(f'{rand_init_i = }')
                confidences = []
                freqs = []
                epochs = []
                for epoch in tqdm(EPOCHS(experiment)):
                    try:
                        nitf = loadNITFForEval(
                            EXP_PATH, experiment.datasetDef, 
                            group, rand_init_i, epoch, 
                        )
                    except FileNotFoundError as e:
                        print(epoch, e)
                        break
                    confidences.append(nitf.dredge_confidence[SELECT_PAGE, :])
                    freqs      .append(nitf.dredge_freq      [SELECT_PAGE].item() * FREQ_SCALE)
                    epochs.append(epoch)
                print('final f0 is', freqs[-1])
                confidences = torch.stack(confidences, dim=0).T
                fig, axes = plt.subplots(2, sharex=True)
                axes: List[plt.Axes]
                axes[0].plot(epochs, freqs, label='freq')
                for i, mult in enumerate(DREDGE_MULT):
                    kw = {}
                    if i == DREDGE_RADIUS:
                        kw['label'] = 'YIN'
                    axes[0].axhline(
                        yin_f0 * mult, c=C[i], 
                        linewidth=.5, 
                        **kw, 
                    )

                    axes[1].plot(
                        epochs, confidences[i, :], 
                        'xv^s<>.'[i], 
                        markersize=4, markerfacecolor='none', 
                        markeredgewidth=.5, 
                        c=C[i], 
                        label=f'{mult:.2f}', 
                    )
                axes[0].legend()
                axes[1].legend()
                axes[1].set_xlabel('Epoch')
                axes[0].set_ylabel('Freq (Hz)')
                axes[1].set_ylabel('Confidence')
                plt.show()

if __name__ == '__main__':
    main()

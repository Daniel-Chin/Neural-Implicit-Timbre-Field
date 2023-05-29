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
    pass

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval
from dredge import *

from workspace import (
    EXP_PATH, EPOCHS, SELECT_PAGE, SELECT_GROUPS, 
)

C = colorLadder(DREDGE_LEN)
C[DREDGE_RADIUS] = 'k'

def main():
    print(f'{SELECT_PAGE = }')
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        assert next(iter(EPOCHS(experiment))) == 0
        dataset: MyDataset = experiment.dataset
        
        truth_f0s = []
        for f0_track in dataset.f0_tracks:
            truth_f0s.append(f0_track[SELECT_PAGE].item())
        
        for group in groups[SELECT_GROUPS]:
            group: ExperimentGroup
            print(group.name())
            for rand_init_i in range(n_rand_inits):
            # rand_init_i = 0
            # if True:
                print(f'{rand_init_i = }')
                for nitf_i in range(group.hyperParams.n_nifs):
                    print(f'{nitf_i = }')
                    confidences = []
                    freqs = []
                    epochs = []
                    for epoch in tqdm(EPOCHS(experiment)):
                        try:
                            nitfs = loadNITFForEval(
                                EXP_PATH, experiment.datasetDef, 
                                group, rand_init_i, epoch, 
                            )
                        except FileNotFoundError as e:
                            print(epoch, e)
                            break
                        nitf = nitfs[nitf_i]
                        confidences.append(nitf.dredge_confidence[SELECT_PAGE, :])
                        freqs.append(freqDenorm(nitf.dredge_freq[SELECT_PAGE].item()))
                        epochs.append(epoch)
                    print('final f0 is', freqs[-1])
                    confidences = torch.stack(confidences, dim=0).T
                    fig, axes = plt.subplots(2, sharex=True)
                    axes: List[plt.Axes]
                    axes[0].plot(epochs, freqs, label='freq')
                    for i, mult in enumerate(DREDGE_MULT):
                        kw = {}
                        if i == DREDGE_RADIUS:
                            kw['label'] = 'Truth'
                        for truth_f0 in truth_f0s:
                            axes[0].axhline(
                                truth_f0 * mult, c=C[i], 
                                linewidth=.5, 
                                **kw, 
                            )
                            kw.clear()

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

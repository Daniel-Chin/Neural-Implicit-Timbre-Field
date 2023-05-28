from os import path
from typing import *

import torch
from matplotlib import pyplot as plt
from torchWork import DEVICE

from shared import *
from nitf import NITF
from dataset import MyDataset
from dataset_definitions import DatasetDefinition, sample_page_lookup
from dredge import *

Y_LIM = 1.0
SCALE_CONFIDENCE = .5

def logSamplePage(
    epoch, nitfs: List[NITF], hParams, 
    dataset: MyDataset, 
    datasetDef: DatasetDefinition, 
    save_path, forwardF0IsLatent, 
):
    for nitf_i, nitf in enumerate(nitfs):
        fig = plt.figure(figsize=(10, 5), num=1, clear=True)
        ax = fig.add_subplot()
        page_i = sample_page_lookup[datasetDef.wav_path]
        spectrum = dataset.X[page_i, :].cpu()
        spectrum_hat: torch.Tensor = forwardF0IsLatent(
            nitf, dataset, hParams, torch.tensor([page_i]), 
        )[0, :].cpu()
        ax.set_xlabel('Freq (Hz)')
        ax.plot(dataset.freqs, spectrum, label='Ground truth')
        ax.plot(dataset.freqs, spectrum_hat, label='Recon')
        X = torch.linspace(
            pitch2freq(36), 
            NYQUIST, 
            1000, 
            device=DEVICE, 
        )
        X = X.unsqueeze(1)
        Y = nitf.forward(freqNorm(X))
        ax.plot(X.cpu(), Y.cpu(), label='NITF')
        f0 = freqDenorm(nitf.dredge_freq[page_i])
        f0s = f0.cpu() * DREDGE_MULT.cpu()
        ax.vlines(
            f0s, ymin=0, 
            ymax = nitf.dredge_confidence[page_i, :].cpu().abs() * SCALE_CONFIDENCE, 
            colors='k', label='Dredge', 
        )
        ax.legend()
        ax.set_ylim(0, Y_LIM)
        ax.set_title(f'epoch {epoch}')
        fig.savefig(path.join(
            save_path, 
            f'sample_page_nitf_{nitf_i}_epoch_{epoch}.png', 
        ))

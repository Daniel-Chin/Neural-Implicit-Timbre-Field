from os import path

import torch
from matplotlib import pyplot as plt

from shared import *
from nitf import NITF
from dataset import MyDataset
from dataset_definitions import DatasetDefinition, sample_page_lookup
from dredge import *

SCALE_CONFIDENCE = .02

def logSamplePage(
    epoch, nitf: NITF, hParams, 
    dataset: MyDataset, 
    datasetDef: DatasetDefinition, 
    save_path, forwardF0IsLatent, 
):
    fig = plt.figure(figsize=(10, 5), num=1, clear=True)
    ax = fig.add_subplot()
    page_i = sample_page_lookup[datasetDef.wav_path]
    spectrum = dataset.X[page_i, :]
    spectrum_hat: torch.Tensor = forwardF0IsLatent(
        nitf, dataset, hParams, torch.tensor([page_i]), 
    )[0, :]
    ax.set_xlabel('Freq (Hz)')
    ax.plot(dataset.freqs, spectrum, label='Ground truth')
    ax.plot(dataset.freqs, spectrum_hat, label='Recon')
    X = torch.linspace(
        pitch2freq(36), 
        NYQUIST, 
        1000, 
    )
    X = X.unsqueeze(1)
    Y = nitf.forward(freqNorm(X))
    ax.plot(X, Y, label='NITF')
    f0 = freqDenorm(nitf.dredge_freq[page_i])
    f0s = f0 * DREDGE_MULT
    ax.vlines(
        f0s, ymin=0, 
        ymax = nitf.dredge_confidence[page_i, :].abs() * SCALE_CONFIDENCE, 
        colors='k', label='Dredge', 
    )
    ax.legend()
    # ax.set_ylim(0, .04)
    ax.set_title(f'epoch {epoch}')
    fig.savefig(path.join(
        save_path, 
        f'sample_page_epoch_{epoch}.png', 
    ))

from os import path
from typing import *

import torch
from matplotlib import pyplot as plt
from torchWork import DEVICE

from shared import *
from nitf import NITF
from dataset import MyDataset
from dataset_definitions import DatasetDefinition, samplePageOf
from dredge import *

def logSamplePage(
    epoch, nitfs: List[NITF], hParams, 
    dataset: MyDataset, 
    datasetDef: DatasetDefinition, 
    save_path, forwardF0IsLatent, 
):
    page_i = samplePageOf(datasetDef)
    spectrum = dataset.X[page_i, :].cpu()
    spectrum_hats = []
    def OneNitf(nitf_i, nitf: NITF):
        spectrum_hat: torch.Tensor = forwardF0IsLatent(
            nitf, dataset, hParams, torch.tensor([page_i]), 
        )[0, :].cpu()
        spectrum_hats.append(spectrum_hat)
        yield
        others = sum_spectrum_hat - spectrum_hat
        residual = spectrum - others
        fig = plt.figure(figsize=(10, 5), num=1, clear=True)
        ax = fig.add_subplot()
        ax.set_xlabel('Freq (Hz)')
        ax.plot(dataset.freqs, residual, label='Residual ground truth')
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
        y_lim = dataset.mean_amp * WINDOW_ENERGY * 8
        ax.vlines(
            f0s, ymin=0, 
            ymax = nitf.dredge_confidence[page_i, :].cpu().abs() * y_lim * .5, 
            colors='k', label='Dredge', 
        )
        ax.legend()
        ax.set_ylim(0, y_lim)
        ax.set_title(f'epoch {epoch}')
        fig.savefig(path.join(
            save_path, 
            f'sample_page_nitf_{nitf_i}_epoch_{epoch}.png', 
        ))
    ones = []
    for nitf_i, nitf in enumerate(nitfs):
        one = OneNitf(nitf_i, nitf)
        ones.append(one)
        next(one)
    sum_spectrum_hat = torch.stack(spectrum_hats, dim=0).sum(dim=0)
    for one in ones:
        try:
            next(one)
        except StopIteration:
            pass
        else:
            assert False

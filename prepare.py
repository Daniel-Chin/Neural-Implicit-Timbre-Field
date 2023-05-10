from __future__ import annotations

from typing import *

import librosa
import numpy as np
from numpy.fft import rfft
from matplotlib import pyplot as plt
import scipy
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from yin import yin
from harmonicSynth import HarmonicSynth, Harmonic

PAGE_LEN = 512
SR = 11025
DTYPE = np.float32
N_HARMONICS = 100

TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, PAGE_LEN)
SPECTRUM_SIZE = PAGE_LEN // 2 + 1
NYQUIST = SR // 2

def pitch2freq(pitch):
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def freq2pitch(f):
    return np.log(f) * 17.312340490667562 - 36.37631656229591

def pagesOf(signal):
    for i in range(0, signal.size - PAGE_LEN + 1, PAGE_LEN):
        yield signal[i : i + PAGE_LEN]

def plotUnstretchedPartials(f0, n_partials = 14, color = 'r', alpha = .3):
    for i in range(1, n_partials + 1):
        freq = f0 * i
        plt.axvline(x = freq, color = color, alpha = alpha)

def prepare(song_filename):
    y, sr = librosa.load(song_filename, sr=SR)
    assert sr == SR

    f0s, timbres, amps = getStreams(y)

    dataset = MyDataset(f0s, timbres, amps)
    print('dataset ok')

    return f0s, timbres, amps, dataset

class MyDataset(Dataset):
    def __init__(self, f0s, timbres, amps) -> None:
        super().__init__()

        I = []
        X = []
        Y = []
        for page_i, (f0, harmonics, amp) in tqdm([*enumerate(
            zip(f0s, timbres, amps), 
        )], desc='prep data'):
            page_X = []
            for harmonic in harmonics:
                page_X.append(torch.tensor((
                    harmonic.freq, f0, amp, 
                )))
                Y.append(harmonic.mag)
                I.append(page_i)
            page_X = torch.stack(page_X)
            # X.append(torch.concat((
            #     page_X, vowel_emb.unsqueeze(0).repeat(len(harmonics), 1), 
            # ), dim=1))
            X.append(page_X)
        X = torch.concat(X, dim=0).float()
        Y = torch.tensor(Y).float()
        I = torch.tensor(I, dtype=torch.long)

        self.X_mean = X.mean(dim=0)
        X = X - self.X_mean
        self.X_std = X.std(dim=0)
        X = X / self.X_std

        self.Y_mean = Y.mean(dim=0)
        Y = Y - self.Y_mean
        self.Y_std = Y.std(dim=0)
        Y = Y / self.Y_std

        self.X = X
        self.Y = Y
        self.I = I
    
    def transformX(self, x):
        return (x - self.X_mean) / self.X_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (
            self.X[index, :], 
            self.Y[index], 
            self.I[index], 
        )

def getStreams(y):
    f0s = []
    amps = []
    timbres: List[List[Harmonic]] = []

    for page_i, page in tqdm(
        [*enumerate(pagesOf(y))], 
        desc='extract timbre', 
    ):
        # spectrum = spectrogram[:, page_i]
        spectrum = np.abs(rfft(page * HANN)) / PAGE_LEN
        f0 = yin(
            page, SR, PAGE_LEN, 
            fmin=pitch2freq(36), 
            fmax=pitch2freq(84), 
        )
        harmonics_f = np.arange(f0, NYQUIST, f0)
        assert harmonics_f.size < N_HARMONICS
        harmonics_a_2 = np.zeros((harmonics_f.size, ))
        spectrum_2 = np.square(spectrum)
        bins_taken = 0
        for partial_i, freq in enumerate(harmonics_f):
            mid_f_bin = round(freq * PAGE_LEN / SR)
            for offset in range(-2, 3):
                try:
                    harmonics_a_2[partial_i] += spectrum_2[
                        mid_f_bin + offset
                    ]
                except IndexError:
                    pass
                else:
                    bins_taken += 1
        mean_bin_noise = (spectrum_2.sum() - harmonics_a_2.sum()) / (
            len(spectrum_2) - bins_taken
        )
        harmonics_a_2[harmonics_a_2 < 2 * mean_bin_noise] = 0
        harmonics_a = np.sqrt(harmonics_a_2)

        harmonics = [
            Harmonic(f, a) for (f, a) in zip(
                harmonics_f, 
                harmonics_a, 
            )
        ]
        freq = harmonics_f[-1]
        for _ in range(len(harmonics), N_HARMONICS):
            freq += f0
            harmonics.append(Harmonic(freq, 0))
        f0s.append(f0)
        timbres.append(harmonics)
        amps.append(np.sqrt(spectrum_2.sum()))
    
    return f0s, timbres, amps

class NITF(nn.Module):
    def __init__(self, width, depth, n_vowel_dims, n_pages) -> None:
        super().__init__()
        layers = []
        in_dim = 3 + n_vowel_dims
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.ReLU())
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.sequential = nn.Sequential(*layers)

        self.vowel_embs = torch.zeros(
            (n_pages, n_vowel_dims), 
            requires_grad=True, 
        )
    
    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

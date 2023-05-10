from __future__ import annotations

print('importing py...')
from typing import *
from os import path
from itertools import count

print('importing 3rd-party...')
import librosa
import numpy as np
from numpy.fft import rfft
from numpy import pi
from matplotlib import pyplot as plt
import scipy
from scipy.signal import stft
from scipy.io import wavfile
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

print('importing dan\'s...')
from yin import yin
from harmonicSynth import HarmonicSynth, Harmonic
from blindDescend import blindDescend

from exp import getTrainers

LR = 1e-2

PAGE_LEN = 512
SR = 11025
DTYPE = np.float32
N_HARMONICS = 100

TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, PAGE_LEN)
SPECTRUM_SIZE = PAGE_LEN // 2 + 1
NYQUIST = SR // 2

def sino(freq, length):
    return np.sin(np.arange(length) * freq * TWO_PI / SR)

def findPeaks(energy):
    slope = np.sign(energy[1:] - energy[:-1])
    extrema = slope[1:] - slope[:-1]
    return np.argpartition(
        (extrema == -2) * energy[1:-1], - N_HARMONICS,
    )[- N_HARMONICS:] + 1

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(IMAGINARY_LADDER * freq_bin))) / PAGE_LEN

def refineGuess(guess, signal):
    def loss(x):
        if x < 0:
            return 0
        return - sft(signal, x)
    freq_bin, loss = blindDescend(loss, .01, .4, guess)
    return freq_bin * SR / PAGE_LEN, - loss
    
def spectrum(signal, trim = 130):
    energy = np.abs(rfft(signal * HANN))
    X = np.linspace(0, SR / 2, len(energy))
    plt.plot(
        X     [:trim], 
        energy[:trim], 
    )
    plt.xlabel('freq (Hz)')

def spectrogram(signal, **kw):
    f, t, Zxx = stft(signal, fs=SR, **kw)
    plt.pcolormesh(t, f, np.abs(Zxx))

def concatSynth(synth, harmonics, n):
    buffer = []
    for i in range(n):
        synth.eat(harmonics)
        buffer.append(synth.mix())
    return np.concatenate(buffer)

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

class Trainer:
    def __init__(
        self, width, depth, n_vowel_dims, 
        batch_size, 
        n_pages, 
    ) -> None:
        self.width: int = width
        self.depth: int = depth
        self.n_vowel_dims: int = n_vowel_dims
        self.batch_size: int = batch_size
        self.losses = []

        self.nitf: NITF
        self.vowel_emb = newVowelEmb(n_pages, n_vowel_dims)
        self.trainIter: Generator
    
    def ready(self, dataset):
        self.nitf = NITF(
            self.width, 
            self.depth, 
            self.n_vowel_dims, 
        )
        self.trainIter = Train(
            self.nitf, self.vowel_emb, dataset, 
            self.batch_size, 
        )
        return self

def main():
    raw = []

    # y, sr = librosa.load('dan.wav', sr=SR)
    # assert sr == SR
    # raw.append(y)

    y, sr = librosa.load('yanhe.wav', sr=SR)
    assert sr == SR
    raw.append(y)

    f0s, timbres, amps = getStreams(y)

    dataset = MyDataset(f0s, timbres, amps)
    print('dataset ok')

    trainers = getTrainers(len(f0s), dataset)

    try:
        for epoch in count():
            print(f'{epoch = }', end=', ')
            for trainer_i, trainer in enumerate(trainers):
                _, _, loss = next(trainer.trainIter)
                trainer.losses.append(loss.item())
                print(loss.item(), end=', ')
                torch.save(trainer.nitf.state_dict(), path.join(
                    './checkpoints', 
                    f'a_{trainer_i}_{trainer.width}_{epoch}.pt', 
                ))
            print()
    except KeyboardInterrupt:
        pass

    C = {
        64: 'r', 
        128: 'g', 
        256: 'b', 
    }
    for trainer in trainers:
        plt.plot(
            trainer.losses, c=C[trainer.width], 
            label=f'width={trainer.width}', 
        )
    plt.legend()
    plt.show()

def newVowelEmb(n_pages, n_vowel_dims):
    return torch.zeros(
        (n_pages, n_vowel_dims), 
        requires_grad=True, 
    )

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
    def __init__(self, width, depth, n_vowel_dims) -> None:
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
    
    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

def Train(nitf: NITF, vowel_embs, dataset, batch_size):
    dataLoader = DataLoader(dataset, batch_size, shuffle=True)
    
    optim = torch.optim.Adam([
        *nitf.parameters(), 
        vowel_embs, 
    ], LR)

    while True:
        nitf.train()
        losses = []
        _iter = dataLoader
        # _iter = tqdm([*_iter], desc='batches')
        for x, y, page_i in _iter:
            x_vowel = torch.concat((
                x, vowel_embs[page_i], 
            ), dim=1)
            # print('forward...')
            y_hat = nitf.forward(x_vowel)
            # print('mse...')
            loss = F.mse_loss(y_hat[:, 0], y)
            # print('zero_grad...')
            optim.zero_grad()
            # print('backward...')
            loss.backward()
            # print('step...')
            optim.step()
            # print('loss...')
            losses.append(loss.detach())
        yield nitf, vowel_embs, torch.tensor(losses).mean()

def selfRecon(f0s, timbres):
    hS = HarmonicSynth(
        N_HARMONICS, SR, PAGE_LEN, DTYPE, True, True, 
    )
    buffer = []

    for i, _ in tqdm([*enumerate(f0s)], desc='recon'):
        harmonics = timbres[i]
        # harmonics = []
        # for j in range(N_HARMONICS):
        #     to_min = []
        #     for offset in (-1, 2):
        #         try:
        #             to_min.append(timbres[i + offset][j].mag)
        #         except IndexError:
        #             pass
        #     mag = min(to_min)
        #     harmonics.append(Harmonic(timbres[i][j].freq, mag))
    
        hS.eat(harmonics)
        buffer.append(hS.mix())

    recon = np.concatenate(buffer)

    wavfile.write('self_recon.wav', SR, recon)

if __name__ == '__main__':
    main()

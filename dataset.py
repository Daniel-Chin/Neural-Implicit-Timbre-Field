from typing import *

import numpy as np
from numpy.fft import rfft
from scipy.signal import stft
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from import_dan_py import ImportDanPy
with ImportDanPy():
    from harmonicSynth import Harmonic
    from yin import yin

from shared import *
from dataset_definitions import DatasetDefinition

class MyDataset(Dataset):
    def __init__(self, datasetDef: DatasetDefinition) -> None:
        super().__init__()

        self.datasetDef = datasetDef

        y, sr = librosa.load(datasetDef.wav_path, sr=SR)
        assert sr == SR

        if datasetDef.is_f0_latent:
            freqs, times, Zxx = stft(
                y, fs=SR, nperseg=PAGE_LEN, 
            )
            self.freqs = torch.tensor(freqs)
            self.times = torch.tensor(times)
            self.n_freq_bins = len(freqs)
            self.one_over_freq_bin = 1 / self.freqs[1]
            X = torch.tensor(np.abs(Zxx))
            self.n_pages = len(self.times)

            self.X_std = X.std(dim=0)
            X = X / self.X_std

            self.X = X.contiguous()
            self.PAGE_I = torch.arange(
                0, self.n_pages, dtype=torch.long, 
            ).contiguous()
        else:
            f0s, timbres, amps = self.getStreams(y)
            self.n_pages = len(f0s)
            self.initF0IsLatent(f0s, timbres, amps)
        
        print('dataset ok')

    def getStreams(self, y):
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

    def initF0IsLatent(
        self, f0s, timbres: List[List[Harmonic]], amps, 
    ):
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
        X = torch.concat(X, dim=0).float().contiguous()
        Y = torch.tensor(Y).float().contiguous()
        I = torch.tensor(I, dtype=torch.long).contiguous()

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
    
    def retransformY(self, y):
        return y * self.Y_std + self.Y_mean

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.datasetDef.is_f0_latent:
            return (
                self.X[index, :], 
                self.I[index], 
            )
        else:
            return (
                self.X[index, :], 
                self.Y[index], 
                self.I[index], 
            )

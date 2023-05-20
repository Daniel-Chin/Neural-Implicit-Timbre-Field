from typing import *

import numpy as np
from numpy.fft import rfft
from scipy.signal import stft
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchWork import DEVICE, HAS_CUDA

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

        print('read wav...')
        y, sr = librosa.load(datasetDef.wav_path, sr=SR)
        print('read wav ok')
        assert sr == SR
        # for debug
        # y = y[:round(len(y) * .1)]; assert not HAS_CUDA
        self.wav = y

        if datasetDef.is_f0_latent:
            # print('stft...')
            freqs, times, Zxx = stft(
                y, fs=SR, nperseg=PAGE_LEN, 
            )
            # print('stft ok')
            self.freqs = torch.tensor(freqs).float()
            self.times = torch.tensor(times).float()
            self.n_freq_bins = len(freqs)
            self.one_over_freq_bin = 1 / self.freqs[1]
            X = torch.tensor(np.abs(Zxx)).T.float().contiguous()
            self.n_pages = len(self.times)

            X = X / X.std(dim=0)

            self.X = X.to(DEVICE).contiguous()
            self.I = torch.arange(
                0, self.n_pages, dtype=torch.long, 
            ).to(DEVICE).contiguous()
        else:
            f0s, timbres, amps = self.getStreams(y)
            self.n_pages = len(f0s)
            self.initF0NotLatent(f0s, timbres, amps)
        
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
            spectrum = np.abs(rfft(page * HANN)) / (PAGE_LEN / 2)
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
            n_noise_bins = len(spectrum_2) - bins_taken
            if n_noise_bins != 0:
                mean_bin_noise = (
                    spectrum_2.sum() - harmonics_a_2.sum()
                ) / n_noise_bins
                harmonics_a_2[harmonics_a_2 < 2 * mean_bin_noise] = 0
            harmonics_a = np.sqrt(harmonics_a_2) / WINDOW_ENERGY

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
            amps.append(np.sqrt(np.square(page).mean()))
        
        return f0s, timbres, amps

    def initF0NotLatent(
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
                    freqNorm(harmonic.freq), freqNorm(f0), amp, 
                )))
                Y.append(harmonic.mag)
                I.append(page_i)
            page_X = torch.stack(page_X)
            # X.append(torch.concat((
            #     page_X, vowel_emb.unsqueeze(0).repeat(len(harmonics), 1), 
            # ), dim=1))
            X.append(page_X)
        X = torch.concat(X, dim=0).float().to(DEVICE).contiguous()
        Y = torch.tensor(Y).float().to(DEVICE).contiguous()
        I = torch.tensor(I, dtype=torch.long).to(DEVICE).contiguous()

        self.X = X
        self.Y = Y
        self.I = I
    
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

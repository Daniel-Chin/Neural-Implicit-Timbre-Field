from typing import *

import numpy as np
from numpy.fft import rfft
from scipy.signal import stft
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchWork import DEVICE

from import_dan_py import ImportDanPy
with ImportDanPy():
    from harmonicSynth import Harmonic
    from yin import yin

from shared import *
from dataset_definitions import (
    DatasetDefinition, voiceScaleF0IsLatent as datasetDef, 
)
from manual_fc import ManualFC

class MyDataset(Dataset):
    def __init__(self, datasetDef: DatasetDefinition) -> None:
        super().__init__()

        self.datasetDef = datasetDef

        print('read wav...')
        y, sr = librosa.load(datasetDef.wav_path, sr=SR, mono=True)
        print('read wav ok')
        assert sr == SR
        self.wav = y
        self.mean_amp = np.sqrt(np.square(y).mean())

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
            X = torch.tensor(np.abs(Zxx) / (PAGE_LEN / 2)).T.float().contiguous()
            self.n_pages = len(self.times)

            X = X / X.max()

            self.X = X.to(DEVICE).contiguous()
            self.I = torch.arange(
                0, self.n_pages, dtype=torch.long, 
            ).to(DEVICE).contiguous()

            self.groundTruthF0(datasetDef)
        else:
            f0s, timbres, amps = self.getStreams(y)
            self.n_pages = len(f0s)
            self.initF0NotLatent(f0s, timbres, amps)
        
        print('dataset ok')

    def groundTruthF0(self, datasetDef: DatasetDefinition):
        self.f0_tracks = []
        if datasetDef.urmp_name is None:
            f0_track = []
            for time in self.times:
                n = round(time.item() * SR)
                page = self.wav[n : n + PAGE_LEN]
                page = np.pad(page, (0, PAGE_LEN - len(page)))
                f0_track.append(yin(
                    page, SR, PAGE_LEN, 
                    fmin=pitch2freq(36), 
                    fmax=pitch2freq(84), 
                ))
            self.f0_tracks.append(
                torch.tensor(f0_track).float().to(DEVICE).contiguous()
            )
        else:
            for track_i in range(2):
                f0_track = getUrmpF0(
                    datasetDef.urmp_name, track_i, 
                )(self.times + PAGE_LEN * .5 / SR).float().to(DEVICE).contiguous()
                self.f0_tracks.append(f0_track)

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

def preview():
    from matplotlib import pyplot as plt
    dataset = MyDataset(datasetDef)
    assert datasetDef.is_f0_latent
    for i, time in tqdm([*enumerate(dataset.times)][10:]):
        spectrum = dataset.X[i, :]
        plt.plot(dataset.freqs, spectrum)
        plt.title(f'Page {i} at {time:.3f} sec')
        plt.show()

def getUrmpF0(urmp_name, track_i):
    T = []
    F = []
    with open(urmpF0(urmp_name, track_i), 'r', encoding='utf-8') as f:
        for line in f:
            t, f = line.strip().split('\t')
            T.append(float(t))
            F.append(float(f))
    T.append(T[-1] + 1)
    F.append(F[-1])
    mfc = ManualFC(torch.tensor(T), torch.tensor(F))
    return mfc

if __name__ == '__main__':
    preview()

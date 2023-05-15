from __future__ import annotations

from typing import *
from os import path
from itertools import count

import librosa
import numpy as np
from numpy.fft import rfft
from matplotlib import pyplot as plt
import scipy
from scipy.signal import stft
from scipy.io import wavfile
from tqdm import tqdm

from yin import yin
from harmonicSynth import HarmonicSynth, Harmonic
from blindDescend import blindDescend

from shared import *

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

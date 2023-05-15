from __future__ import annotations

__all__ = [
    'PAGE_LEN', 
    'SR', 
    'DTYPE', 
    'DTYPE_PA', 
    'N_HARMONICS', 
    'TWO_PI', 
    'HANN', 
    'IMAGINARY_LADDER', 
    'SPECTRUM_SIZE', 
    'NYQUIST', 
    'pitch2freq', 
    'freq2pitch', 
    'pagesOf', 
    'plotUnstretchedPartials', 
]

from typing import *

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import pyaudio

PAGE_LEN = 256
SR = 11025
DTYPE = np.float32
DTYPE_PA = pyaudio.paFloat32
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

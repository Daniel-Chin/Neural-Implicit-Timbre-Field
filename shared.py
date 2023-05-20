from __future__ import annotations

__all__ = [
    'PAGE_LEN', 
    'SR', 
    'DTYPE', 
    'N_HARMONICS', 
    'TWO_PI', 
    'HANN', 
    'WINDOW_ENERGY', 
    'IMAGINARY_LADDER', 
    'SPECTRUM_SIZE', 
    'NYQUIST', 
    'pitch2freq', 
    'freq2pitch', 
    'pagesOf', 
    'plotUnstretchedPartials', 
    'freqNorm', 
    'freqDenorm', 
]

from typing import *

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal

PAGE_LEN = 256
SR = 11025
DTYPE = np.float32
N_HARMONICS = 100

TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
WINDOW_ENERGY = np.sqrt(np.square(HANN).sum())
IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, PAGE_LEN)
SPECTRUM_SIZE = PAGE_LEN // 2 + 1
NYQUIST = SR // 2

def pitch2freq(pitch):
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def freq2pitch(f):
    return np.log(f) * 17.312340490667562 - 36.37631656229591

def pagesOf(signal: np.ndarray):
    for i in range(0, signal.size - PAGE_LEN + 1, PAGE_LEN):
        yield signal[i : i + PAGE_LEN]

def plotUnstretchedPartials(f0, n_partials = 14, color = 'r', alpha = .3):
    for i in range(1, n_partials + 1):
        freq = f0 * i
        plt.axvline(x = freq, color = color, alpha = alpha)

def freqNorm(freq):
    return (freq - 200) * 1e-2
# violates DRY. Change together!
def freqDenorm(freq):
    return freq * 1e2 + 200
a = np.random.randn(100)
assert (np.abs(freqNorm(freqDenorm(a)) - a) < 1e-6).all()

from functools import lru_cache

import torch
import numpy as np
from numpy.fft import rfft

from shared import *
from manualFC import ManualFC

def sino(freq, length):
    return np.sin(np.arange(length) * freq * TWO_PI / SR)

def lobeAt(X, f=500, window=HANN):
    freqs = np.linspace(0, SR / 2, PAGE_LEN // 2 + 1)
    freq_bin_width = freqs[1]
    f_i = np.abs(f - freqs).argmin()
    freq = freqs[f_i]
    Y = np.zeros_like(X)
    for i, x in enumerate(X):
        y = sino(freq + x * freq_bin_width, PAGE_LEN)
        energy = np.abs(rfft(y * window)) / (PAGE_LEN / 2)
        Y[i] = energy[f_i]
    return Y

def test():
    from matplotlib import pyplot as plt

    X = np.linspace(-4, +4, 1000)
    Y = lobeAt(X)
    lobe = getLobe()
    Y_ = lobe(torch.tensor(X).unsqueeze(1))
    plt.plot(X, Y, label='truth')
    plt.plot(X, Y_, label='fit', linewidth=.5)

    plt.legend()
    plt.show()

@lru_cache()
def getLobe(resolution=30):
    X = np.linspace(-2, +2, resolution)
    Y = lobeAt(X)
    X = torch.tensor([*X, +2.5])
    Y = torch.tensor([*Y, 0])
    return ManualFC(X, Y)

# @lru_cache()
# def getLobe(resolution=21):   # deprecated due to sharp x=0
#     X = np.linspace(-2, +2, resolution)
#     Y = lobeAt(X)
#     X = torch.tensor(X)
#     Y = torch.tensor(Y)
#     mFC = ManualFC(X[:51], Y[:51])
#     def lobe(x: torch.Tensor):
#         return mFC(- x.abs())
#     return lobe

if __name__ == '__main__':
    test()

print('importing py...')
from typing import *

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

print('importing dan\'s...')
from yin import yin
from harmonicSynth import HarmonicSynth, Harmonic
from blindDescend import blindDescend

PAGE_LEN = 768
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

def main():
    print('main()')

    print('load wav...')
    raw = []

    y, sr = librosa.load('dan.wav', sr=SR)
    assert sr == SR
    raw.append(y)

    # y, sr = librosa.load('yanhe.wav', sr=SR)
    # assert sr == SR
    # raw.append(y)
    
    print('stft...')
    freqs, times, Zxx = stft(
        y, fs=SR, nperseg=PAGE_LEN, 
    )
    spectrogram = np.abs(Zxx)

    f0s = []
    amps = []
    timbres: List[List[Harmonic]] = []

    for page_i, (t, page) in tqdm(
        [*enumerate(zip(times, pagesOf(y)))]
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
    
    # selfRecon(f0s, timbres)



def selfRecon(f0s, timbres):
    hS = HarmonicSynth(
        N_HARMONICS, SR, PAGE_LEN, DTYPE, True, True, 
    )
    buffer = []

    for i, _ in tqdm([*enumerate(f0s)]):
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

main()

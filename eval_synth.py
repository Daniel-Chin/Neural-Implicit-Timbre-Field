from os import path
from typing import *
from functools import lru_cache

import numpy as np
import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk, 
)
from matplotlib.figure import Figure
import matplotlib.animation as animation
import pyaudio

from import_dan_py import ImportDanPy
with ImportDanPy():
    from harmonicSynth import HarmonicSynth, Harmonic
    from selectAudioDevice import selectAudioDevice

from shared import *
from exp_group import ExperimentGroup
from nitf import NITF
from dataset import MyDataset
from load_for_eval import loadNITFForEval

from workspace import EXP_PATH, EPOCHS

PLOT_MAX_FPS = 10
PLOT_RESOLUTION = 200
POINT_RADIUS = 2
VOWEL_SPACE_Z_RADIUS = 3

DTYPE_PA = pyaudio.paFloat32
MASTER_VOLUME = 1e-3

plotVowels = None   # violates MVC
anim = None

class LeftFrame(tk.Frame):
    def __init__(
        self, parent, groups: List[ExperimentGroup], 
        refreshNITF, 
        group_selection, 
    ) -> None:
        super().__init__(parent)

        for i, group in enumerate(groups):
            rB = tk.Radiobutton(
                self, text=group.name(), 
                variable=group_selection, value=i, 
                command=refreshNITF, 
            )
            rB.pack(anchor=tk.W)

class RightFrame(tk.Frame):
    class SpinsFrame(tk.Frame):
        def __init__(
            self, parent, refreshNITF, rand_init_i, epoch, 
        ) -> None:
            super().__init__(parent)

            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            self.columnconfigure(2, weight=0)
            self.columnconfigure(3, weight=1)
            self.rowconfigure(0, weight=1)

            tk.Label(self, text='rand_init_i:').grid(
                row=0, column=0, sticky=tk.NSEW, 
            )
            tk.Spinbox(
                self, textvariable=rand_init_i, 
                from_=-1, to=1e8, 
                command=refreshNITF, 
            ).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )

            tk.Label(self, text='epoch:').grid(
                row=0, column=2,  sticky=tk.NSEW, 
            )
            tk.Spinbox(
                self, textvariable=epoch, 
                from_=0, to=1e8, 
                command=refreshNITF, 
            ).grid(
                row=0, column=3, sticky=tk.NSEW, 
            )
    
    class SquaresFrame(tk.Frame):
        class SpectralEnvelopeFrame(tk.Frame):
            def __init__(
                self, parent, 
                groups: List[ExperimentGroup], 
                group_selection: tk.IntVar, 
                refreshNITF, 
                pitch: tk.DoubleVar, amp: tk.DoubleVar, 
                vowel_emb_zscore_0: tk.DoubleVar, 
                vowel_emb_zscore_1: tk.DoubleVar, 
                nitfContainer: List[NITF], 
                dataset: MyDataset, 
            ) -> None:
                super().__init__(parent)

                self.groups = groups
                self.group_selection = group_selection
                self.refreshNITF = refreshNITF
                self.pitch = pitch
                self.amp = amp
                self.vowel_emb_zscore_0 = vowel_emb_zscore_0
                self.vowel_emb_zscore_1 = vowel_emb_zscore_1
                self.nitfContainer = nitfContainer
                self.dataset = dataset

                self.fig = Figure(
                    figsize=(.2, .1), dpi=100, 
                )
                figure_canvas = FigureCanvasTkAgg(self.fig, self)
                # NavigationToolbar2Tk(figure_canvas, self)
                self.canvas = figure_canvas.get_tk_widget()
                self.canvas.pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True, 
                )
                ax = self.fig.add_subplot()
                X = np.linspace(0, NYQUIST, PLOT_RESOLUTION)
                self.line2D = ax.plot(X, np.ones_like(X))[0]
                self.X = torch.Tensor(X)
                ax.axhline(y=0, c='k', linewidth=.5)
                ax.set_xlabel('frequency (Hz)')
                ax.set_ylabel('envelope')
                # ax.set_ylim(bottom=-.001, top=.01)
                self.ax = ax

                global anim
                anim = animation.FuncAnimation(
                    self.fig, self.replot, interval=1000 / PLOT_MAX_FPS, 
                )

                # self.throttle = 0
            
            def replot(self, _):
                # if time() < self.throttle:
                #     return

                if self.nitfContainer[0] is None:
                    return
                f0 = pitch2freq(self.pitch.get())
                mag = inference(
                    self.X, 
                    f0, 
                    self.nitfContainer, self.groups, 
                    self.dataset, self.group_selection, 
                    self.amp, 
                    self.vowel_emb_zscore_0, 
                    self.vowel_emb_zscore_1, 
                )
                self.line2D.set_ydata(mag)
                self.ax.set_ylim(bottom=mag.min()-.01, top=mag.max()+.01)

                self.fig.tight_layout()
                # self.throttle = time() + PLOT_MAX_SPF
        
        class TouchPad(tk.Canvas):
            def __init__(
                self, parent: tk.Widget, 
                nitfContainer: List[NITF], 
            ) -> None:
                super().__init__(
                    parent, background='#dddddd', 
                )

                self.parent = parent
                self.nitfContainer = nitfContainer

                global plotVowels
                plotVowels = self.plotVowels
            
            def plotVowels(self):
                width  = self.parent.winfo_width() * .5
                height = self.parent.winfo_height()
                self.configure(width=width, height=height)
                wh_half = torch.tensor([width, height]) * .5
                nitf = self.nitfContainer[0]
                ves = nitf.vowel_embs
                ves -= ves.mean()
                ves /= ves.std() * VOWEL_SPACE_Z_RADIUS
                ves = ves.clip(min=-1, max=+1)
                ves *= wh_half
                ves += wh_half
                self.delete('all')
                self.create_text(
                    width * .5, height * .5, 
                    text='touch pad', 
                )
                for i in range(ves.shape[0]):
                    ve = [x.item() for x in ves[i, :]]
                    ve.append(0) # in case 1-d ve
                    self.create_rectangle(
                        ve[0] - POINT_RADIUS, 
                        ve[1] - POINT_RADIUS, 
                        ve[0] + POINT_RADIUS, 
                        ve[1] + POINT_RADIUS, 
                    )
        
        def __init__(
            self, parent, groups, group_selection, 
            refreshNITF, 
            pitch, amp, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, dataset, 
        ) -> None:
            super().__init__(parent)

            self.refreshNITF = refreshNITF
            self.vowel_emb_zscore_0 = vowel_emb_zscore_0
            self.vowel_emb_zscore_1 = vowel_emb_zscore_1

            self.columnconfigure(0, weight=0)
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

            touchPad = self.TouchPad(
                self, nitfContainer, 
            )
            touchPad.grid(
                row=0, column=0, sticky=tk.NSEW, 
            )
            touchPad.bind('<Motion>', self.onMotion)

            self.SpectralEnvelopeFrame(
                self, groups, group_selection, 
                refreshNITF, pitch, amp, 
                vowel_emb_zscore_0, vowel_emb_zscore_1, 
                nitfContainer, dataset, 
            ).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )
        
        def onMotion(self, event: tk.Event):
            x = event.x / self.winfo_width()
            y = event.y / self.winfo_height()
            self.vowel_emb_zscore_0.set((x - .5) * 2 * VOWEL_SPACE_Z_RADIUS)
            self.vowel_emb_zscore_1.set((y - .5) * 2 * VOWEL_SPACE_Z_RADIUS)
            # self.refreshNITF()
    
    class PitchFrame(tk.Frame):
        def __init__(self, parent, refreshNITF, pitch) -> None:
            super().__init__(parent)

            tk.Label(self, text="pitch:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=pitch, 
                # command=refreshNITF, 
                from_=24, to=108, 
                resolution=0.01, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    class AmpFrame(tk.Frame):
        def __init__(self, parent, refreshNITF, amp) -> None:
            super().__init__(parent)

            tk.Label(self, text="amp:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=amp, 
                # command=refreshNITF, 
                from_=0, to=1e-1, 
                resolution=1e-3, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    def __init__(
        self, parent, groups, group_selection, 
        refreshNITF, pitch, amp, 
        rand_init_i, epoch, 
        vowel_emb_zscore_0, vowel_emb_zscore_1, 
        nitfContainer, dataset, 
    ) -> None:
        super().__init__(parent)

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        self.columnconfigure(0, weight=1)

        self.SpinsFrame(
            self, refreshNITF, rand_init_i, epoch, 
        ).grid(
            row=0, column=0, sticky=tk.NSEW, 
        )
        self.SquaresFrame(
            self, groups, group_selection, 
            refreshNITF, pitch, amp, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, dataset, 
        ).grid(
            row=1, column=0, sticky=tk.NSEW, 
        )
        self.PitchFrame(
            self, refreshNITF, pitch, 
        ).grid(
            row=2, column=0, sticky=tk.NSEW, 
        )
        self.AmpFrame(
            self, refreshNITF, amp, 
        ).grid(
            row=3, column=0, sticky=tk.NSEW, 
        )

def initRoot(
    root: tk.Tk, groups, 
    refreshNITF, 
    group_selection, pitch, amp, rand_init_i, epoch, 
    vowel_emb_zscore_0, vowel_emb_zscore_1, 
    nitfContainer, dataset, 
) -> None:
    root.title('Eval NITF')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=5)
    root.rowconfigure(0, weight=1)
    
    leftFrame = LeftFrame(
        root, groups, refreshNITF, group_selection, 
    )
    leftFrame.grid(
        row=0, column=0, sticky=tk.NSEW, 
    )

    rightFrame = RightFrame(
        root, groups, group_selection, 
        refreshNITF, pitch, amp, rand_init_i, epoch, 
        vowel_emb_zscore_0, vowel_emb_zscore_1, 
        nitfContainer, dataset, 
    )
    rightFrame.grid(
        row=0, column=1, sticky=tk.NSEW, 
    )

class AudioStreamer:
    def __init__(
        self, nitfContainer: List[NITF], 
        groups: List[ExperimentGroup], 
        group_selection: tk.IntVar, 
        pitch: tk.DoubleVar, amp: tk.DoubleVar, 
        vowel_emb_zscore_0: tk.DoubleVar, 
        vowel_emb_zscore_1: tk.DoubleVar, 
        dataset: MyDataset, 
    ) -> None:
        self.hS = HarmonicSynth(
            N_HARMONICS, SR, PAGE_LEN, DTYPE, True, True, 
        )
        self.nitfContainer = nitfContainer
        self.groups = groups
        self.group_selection = group_selection
        self.pitch = pitch
        self.amp = amp
        self.vowel_emb_zscore_0 = vowel_emb_zscore_0
        self.vowel_emb_zscore_1 = vowel_emb_zscore_1
        self.dataset = dataset

    def nextPageOut(self, in_data, frame_count, time_info, status):
        with torch.no_grad():
            assert frame_count == PAGE_LEN

            f0 = pitch2freq(self.pitch.get())
            freqs = torch.arange(1, N_HARMONICS + 1, dtype=torch.float32) * f0
            mag = inference(
                freqs, 
                f0, 
                self.nitfContainer, self.groups, 
                self.dataset, self.group_selection, 
                self.amp, 
                self.vowel_emb_zscore_0, 
                self.vowel_emb_zscore_1, 
            ).clip(min=0)
            harmonics = []
            for partial_i in range(N_HARMONICS):
                harmonics.append(Harmonic(
                    freqs[partial_i].item(), 
                    mag[partial_i].item(), 
                ))
            self.hS.eat(harmonics)
            page_out = self.hS.mix() * MASTER_VOLUME
            page_max = np.max(np.abs(page_out))
            if page_max > .7:
                print('Warning: audio max is large =', page_max)
            return page_out, pyaudio.paContinue

def inference(
    freqs: torch.Tensor, f0, 
    nitfContainer: List[NITF], 
    groups: List[ExperimentGroup], 
    dataset: MyDataset, 
    group_selection: tk.IntVar, 
    amp: tk.DoubleVar, 
    vowel_emb_zscore_0: tk.DoubleVar, 
    vowel_emb_zscore_1: tk.DoubleVar, 
):
    nitf = nitfContainer[0]
    group = groups[group_selection.get()]
    hParams = group.hyperParams
    n_vowel_dims = group.hyperParams.n_vowel_dims
    ve_mean = nitf.vowel_embs.mean(dim=0)
    ve_std  = nitf.vowel_embs.std(dim=0)
    ve = ve_mean
    ve[0] += ve_std[0] * vowel_emb_zscore_0.get()
    if n_vowel_dims >= 2:
        ve[1] += ve_std[1] * vowel_emb_zscore_1.get()
    ve = ve.float()

    X = [freqNorm(freqs)]
    if hParams.nif_sees_f0:
        X.append(freqNorm(torch.ones_like(freqs) * f0))
    if hParams.nif_sees_amp:
        X.append(torch.ones_like(freqs) * amp.get())
    X = torch.stack(X, dim=1)
    if hParams.nif_sees_vowel:
        X = torch.concat((
            X, 
            ve.unsqueeze(0).repeat(X.shape[0], 1), 
        ), dim=1)
    mag = nitf.forward(X)[:, 0]
    return mag

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        @lru_cache()
        def loadNITFWithCache(group_i, rand_init_i, epoch):
            return loadNITFForEval(
                EXP_PATH, experiment.datasetDef, 
                groups[group_i], rand_init_i, epoch, 
            )

        root = tk.Tk()

        group_selection = tk.IntVar(root, 0)
        pitch = tk.DoubleVar(root, 60)
        amp = tk.DoubleVar(root, .01)
        rand_init_i = tk.StringVar(root, 0)
        epoch = tk.StringVar(root, 0)
        vowel_emb_zscore_0 = tk.DoubleVar(root, 0)
        vowel_emb_zscore_1 = tk.DoubleVar(root, 0)
        nitfContainer = [None]

        epoch.set(str(next(iter(EPOCHS(experiment)))))

        def refreshNITF(*_):
            global plotVowels

            cycleIntVar(rand_init_i, 0, n_rand_inits)
            cycleIntVar(epoch, 0, 1e8)

            try:
                nitf = loadNITFWithCache(
                    group_selection.get(), 
                    rand_init_i.get(), 
                    epoch.get(), 
                )
            except FileNotFoundError as e:
                print(e)
                epoch.set('0')
                return refreshNITF()
            nitfContainer[0] = nitf

            if plotVowels is not None:
                plotVowels()

        initRoot(
            root, groups, 
            refreshNITF, 
            group_selection, pitch, amp, rand_init_i, epoch, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, experiment.dataset, 
        )
        refreshNITF()

        pa = pyaudio.PyAudio()
        aS = AudioStreamer(
            nitfContainer, groups, 
            group_selection, pitch, amp, 
            vowel_emb_zscore_0, 
            vowel_emb_zscore_1, 
            experiment.dataset, 
        )
        _, out_i = selectAudioDevice(pa, out_guesses=['Speakers'])
        streamOut = pa.open(
            format = DTYPE_PA, channels = 1, rate = SR, 
            output = True, frames_per_buffer = PAGE_LEN,
            stream_callback = aS.nextPageOut, 
            output_device_index = out_i, 
        )
        streamOut.start_stream()
        try:
            root.mainloop()
        finally:
            global plotVowels
            plotVowels = None   # don't hold onto a widget
            streamOut.close()
            pa.terminate()

def cycleIntVar(intVar: tk.IntVar, start, stop):
    try:
        rand_init_i_int = int(intVar.get())
    except ValueError:
        intVar.set(str(start))
    else:
        if rand_init_i_int < start:
            intVar.set(str(stop - 1))
        if rand_init_i_int >= stop:
            intVar.set(str(start))

if __name__ == '__main__':
    main()

from os import path
from typing import *
from functools import lru_cache

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
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

try:
    from selectAudioDevice import selectAudioDevice
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    print(f'Missing module {module_name}. Please download at')
    print(f'https://github.com/Daniel-Chin/Python_Lib/blob/master/{module_name}.py')
    input('Press Enter to quit...')
    raise e

from prepare import *
from exp_group import ExperimentGroup

from workspace import EXP_PATH

PLOT_MAX_FPS = 10
PLOT_RESOLUTION = 200

# replot = None
anim = None

class LeftFrame(tk.Frame):
    def __init__(
        self, parent, groups: List[ExperimentGroup], 
        mainUpdate, 
        group_selection, 
    ) -> None:
        super().__init__(parent)

        for i, group in enumerate(groups):
            rB = tk.Radiobutton(
                self, text=group.name(), 
                variable=group_selection, value=i, 
                command=mainUpdate, 
            )
            rB.pack(anchor=tk.W)

class RightFrame(tk.Frame):
    class SpinsFrame(tk.Frame):
        def __init__(
            self, parent, mainUpdate, rand_init_i, epoch, 
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
                command=mainUpdate, 
            ).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )

            tk.Label(self, text='epoch:').grid(
                row=0, column=2,  sticky=tk.NSEW, 
            )
            tk.Spinbox(
                self, textvariable=epoch, 
                from_=0, to=1e8, 
                command=mainUpdate, 
            ).grid(
                row=0, column=3, sticky=tk.NSEW, 
            )
    
    class SquaresFrame(tk.Frame):
        class SpectralEnvelopeFrame(tk.Frame):
            def __init__(
                self, parent, 
                groups: List[ExperimentGroup], 
                group_selection: tk.IntVar, 
                mainUpdate, 
                pitch: tk.DoubleVar, amp: tk.DoubleVar, 
                vowel_emb_zscore_0: tk.DoubleVar, 
                vowel_emb_zscore_1: tk.DoubleVar, 
                nitfContainer: List[NITF], 
                dataset: MyDataset, 
            ) -> None:
                super().__init__(parent)

                self.groups = groups
                self.group_selection = group_selection
                self.mainUpdate = mainUpdate
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
                self.ax = ax

                # global replot
                # replot = self.replot
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
                self.ax.set_ylim(bottom=mag.min(), top=mag.max())

                self.fig.tight_layout()
                # self.throttle = time() + PLOT_MAX_SPF
        
        def __init__(
            self, parent, groups, group_selection, 
            mainUpdate, 
            pitch, amp, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, dataset, 
        ) -> None:
            super().__init__(parent)

            self.mainUpdate = mainUpdate
            self.vowel_emb_zscore_0 = vowel_emb_zscore_0
            self.vowel_emb_zscore_1 = vowel_emb_zscore_1

            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

            touchPad = tk.Label(
                self, text='touch pad', background='#dddddd', 
            )
            touchPad.grid(
                row=0, column=0, sticky=tk.NSEW, 
            )
            touchPad.bind('<Motion>', self.onMotion)

            self.SpectralEnvelopeFrame(
                self, groups, group_selection, 
                mainUpdate, pitch, amp, 
                vowel_emb_zscore_0, vowel_emb_zscore_1, 
                nitfContainer, dataset, 
            ).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )
        
        def onMotion(self, event: tk.Event):
            x = event.x / self.winfo_width()
            y = event.y / self.winfo_height()
            self.vowel_emb_zscore_0.set((x - .5) * 4)
            self.vowel_emb_zscore_1.set((y - .5) * 4)
            self.mainUpdate()
    
    class PitchFrame(tk.Frame):
        def __init__(self, parent, mainUpdate, pitch) -> None:
            super().__init__(parent)

            tk.Label(self, text="pitch:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=pitch, 
                command=mainUpdate, 
                from_=24, to=108, 
                resolution=0.01, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    class AmpFrame(tk.Frame):
        def __init__(self, parent, mainUpdate, amp) -> None:
            super().__init__(parent)

            tk.Label(self, text="amp:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=amp, 
                command=mainUpdate, 
                from_=0, to=1e-1, 
                resolution=1e-3, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    def __init__(
        self, parent, groups, group_selection, 
        mainUpdate, pitch, amp, 
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
            self, mainUpdate, rand_init_i, epoch, 
        ).grid(
            row=0, column=0, sticky=tk.NSEW, 
        )
        self.SquaresFrame(
            self, groups, group_selection, 
            mainUpdate, pitch, amp, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, dataset, 
        ).grid(
            row=1, column=0, sticky=tk.NSEW, 
        )
        self.PitchFrame(
            self, mainUpdate, pitch, 
        ).grid(
            row=2, column=0, sticky=tk.NSEW, 
        )
        self.AmpFrame(
            self, mainUpdate, amp, 
        ).grid(
            row=3, column=0, sticky=tk.NSEW, 
        )

def initRoot(
    root: tk.Tk, groups, 
    mainUpdate, 
    group_selection, pitch, amp, rand_init_i, epoch, 
    vowel_emb_zscore_0, vowel_emb_zscore_1, 
    nitfContainer, dataset, 
) -> None:
    root.title('Eval NITF')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=5)
    root.rowconfigure(0, weight=1)
    
    leftFrame = LeftFrame(
        root, groups, mainUpdate, group_selection, 
    )
    leftFrame.grid(
        row=0, column=0, sticky=tk.NSEW, 
    )

    rightFrame = RightFrame(
        root, groups, group_selection, 
        mainUpdate, pitch, amp, rand_init_i, epoch, 
        vowel_emb_zscore_0, vowel_emb_zscore_1, 
        nitfContainer, dataset, 
    )
    rightFrame.grid(
        row=0, column=1, sticky=tk.NSEW, 
    )

def loadNITF(group, rand_init_i, epoch):
    epoch, models = loadLatestModels(
        EXP_PATH, group, rand_init_i, dict(
            nitf=(NITF, 1), 
        ), epoch, verbose=False, 
    )
    nitf = models['nitf'][0]
    nitf.eval()
    nitf.vowel_embs = nitf.get_buffer('saved_vowel_embs')
    return nitf

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
            return self.hS.mix(), pyaudio.paContinue

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
    n_vowel_dims = group.hyperParams.n_vowel_dims
    ve_mean = nitf.vowel_embs.mean(dim=0)
    ve_std  = nitf.vowel_embs.std(dim=0)
    ve = ve_mean
    ve[0] += ve_std[0] * vowel_emb_zscore_0.get()
    if n_vowel_dims >= 2:
        ve[1] += ve_std[1] * vowel_emb_zscore_1.get()
    ve = ve.float()

    X = torch.stack((
        freqs, 
        torch.ones_like(freqs) * f0, 
        torch.ones_like(freqs) * amp.get(), 
    ), dim=1)
    X_vowel = torch.concat((
        dataset.transformX(X), 
        ve.unsqueeze(0).repeat(X.shape[0], 1), 
    ), dim=1)
    mag = dataset.retransformY(
        nitf.forward(X_vowel), 
    )
    return mag

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        @lru_cache()
        def loadNITFWithCache(group_i, rand_init_i, epoch):
            return loadNITF(groups[group_i], rand_init_i, epoch)

        root = tk.Tk()

        group_selection = tk.IntVar(root, 0)
        pitch = tk.DoubleVar(root, 60)
        amp = tk.DoubleVar(root, .01)
        rand_init_i = tk.StringVar(root, 0)
        epoch = tk.StringVar(root, 0)
        vowel_emb_zscore_0 = tk.DoubleVar(root, 0)
        vowel_emb_zscore_1 = tk.DoubleVar(root, 0)
        nitfContainer = [None]

        def mainUpdate(*_):
            # global replot

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
                return mainUpdate()
            nitfContainer[0] = nitf

            # if replot is not None:
            #     replot()

        initRoot(
            root, groups, 
            mainUpdate, 
            group_selection, pitch, amp, rand_init_i, epoch, 
            vowel_emb_zscore_0, vowel_emb_zscore_1, 
            nitfContainer, experiment.dataset, 
        )
        mainUpdate()

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
            # replot = None   # don't hold onto a widget
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

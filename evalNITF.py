from os import path
from typing import *

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk, 
)

from prepare import *
from exp_group import ExperimentGroup

from workspace import EXP_PATH

class LeftFrame(tk.Frame):
    def __init__(
        self, parent, groups: List[ExperimentGroup], 
        resynth, 
        group_selection, 
    ) -> None:
        super().__init__(parent)

        for i, group in enumerate(groups):
            rB = tk.Radiobutton(
                self, text=group.name(), 
                variable=group_selection, value=i, 
                command=resynth, 
            )
            rB.pack(anchor=tk.W)

class RightFrame(tk.Frame):
    class SpinsFrame(tk.Frame):
        def __init__(
            self, parent, resynth, rand_init_i, epoch, 
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
                command=resynth, 
            ).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )

            tk.Label(self, text='epoch:').grid(
                row=0, column=2,  sticky=tk.NSEW, 
            )
            tk.Spinbox(
                self, textvariable=epoch, 
                from_=-1, to=1e8, 
                command=resynth, 
            ).grid(
                row=0, column=3, sticky=tk.NSEW, 
            )
    
    class SquaresFrame(tk.Frame):
        class SpectralEnvelopeFrame(tk.Frame):
            def __init__(self, parent) -> None:
                super().__init__(parent)

                fig = Figure(figsize=(.2, .1), dpi=100)
                figure_canvas = FigureCanvasTkAgg(fig, self)
                NavigationToolbar2Tk(figure_canvas, self)
                ax = fig.add_subplot()
                ax.plot([1,3,2,4,10,5])
                fig.tight_layout()
                figure_canvas.get_tk_widget().pack(
                    side=tk.TOP, fill=tk.BOTH, expand=True, 
                )
        
        def __init__(self, parent) -> None:
            super().__init__(parent)

            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

            touchPad = tk.Label(
                self, text='touch pad', background='#ddffff', 
            )
            touchPad.grid(
                row=0, column=0, sticky=tk.NSEW, 
            )
            touchPad.bind('<Motion>', self.onMotion)

            self.SpectralEnvelopeFrame(self).grid(
                row=0, column=1, sticky=tk.NSEW, 
            )
        
        def onMotion(self, event: tk.Event):
            print(event.x, event.y)
    
    class PitchFrame(tk.Frame):
        def __init__(self, parent, resynth, pitch) -> None:
            super().__init__(parent)

            tk.Label(self, text="pitch:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=pitch, 
                command=resynth, 
                from_=24, to=108, 
                resolution=0.01, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    class AmpFrame(tk.Frame):
        def __init__(self, parent, resynth, amp) -> None:
            super().__init__(parent)

            tk.Label(self, text="amp:").pack(
                side=tk.LEFT, 
            )

            tk.Scale(
                self, variable=amp, 
                command=resynth, 
                from_=0, to=1e-1, 
                resolution=1e-3, 
                orient=tk.HORIZONTAL, 
            ).pack(
                side=tk.LEFT, 
                expand=True, fill=tk.BOTH, 
            )
    
    def __init__(
        self, parent, resynth, pitch, amp, 
        rand_init_i, epoch, 
    ) -> None:
        super().__init__(parent)

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        self.columnconfigure(0, weight=1)

        self.SpinsFrame(
            self, resynth, rand_init_i, epoch, 
        ).grid(
            row=0, column=0, sticky=tk.NSEW, 
        )
        self.SquaresFrame(self).grid(
            row=1, column=0, sticky=tk.NSEW, 
        )
        self.PitchFrame(
            self, resynth, pitch, 
        ).grid(
            row=2, column=0, sticky=tk.NSEW, 
        )
        self.AmpFrame(
            self, resynth, amp, 
        ).grid(
            row=3, column=0, sticky=tk.NSEW, 
        )

def initRoot(
    root: tk.Tk, groups, 
    resynth, 
    group_selection, pitch, amp, rand_init_i, epoch, 
) -> None:
    root.title('Eval NITF')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)
    root.rowconfigure(0, weight=1)
    
    leftFrame = LeftFrame(
        root, groups, resynth, group_selection, 
    )
    leftFrame.grid(
        row=0, column=0, sticky=tk.NSEW, 
    )

    rightFrame = RightFrame(
        root, 
        resynth, pitch, amp, rand_init_i, epoch, 
    )
    rightFrame.grid(
        row=0, column=1, sticky=tk.NSEW, 
    )

def loadNITF(group, rand_init_i, epoch):
    epoch, models = loadLatestModels(
        EXP_PATH, group, rand_init_i, dict(
            nitf=(NITF, 1), 
        ), epoch, 
    )
    nitf = models['nitf'][0]
    nitf.eval()
    return nitf

def main():
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            EXP_PATH, EXPERIMENT_PY_FILENAME, 
        ))
        print(f'{exp_name = }')

        root = tk.Tk()

        group_selection = tk.IntVar(root, 0)
        pitch = tk.DoubleVar(root, 60)
        amp = tk.DoubleVar(root, .001)
        rand_init_i = tk.StringVar(root, 0)
        epoch = tk.StringVar(root, 0)

        def resynth(*args):
            ...

        initRoot(
            root, groups, 
            resynth, 
            group_selection, pitch, amp, rand_init_i, epoch, 
        )
        root.mainloop()

if __name__ == '__main__':
    main()

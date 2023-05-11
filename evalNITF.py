from os import path

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
import tkinter as tk

from prepare import *

from workspace import EXP_PATH

class Window(tk.Tk):
    def __init__(self, ) -> None:
        super().__init__()

        self.title('Eval NITF')

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
        ...

if __name__ == '__main__':
    main()

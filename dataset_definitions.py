from __future__ import annotations

from os import path
from typing import *
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from prepare import *

class DatasetDefinition:
    def __init__(self) -> None:
        self.wav_path: str = None
        self.is_f0_latent: bool = None
        # is f0 and amp latent

dF = DatasetDefinition()
dF.wav_path = './dataset/dan.wav'
dF.is_f0_latent = False
danUseYin = dF

dF = DatasetDefinition()
dF.wav_path = './dataset/yanhe.wav'
dF.is_f0_latent = False
heUseYin = dF

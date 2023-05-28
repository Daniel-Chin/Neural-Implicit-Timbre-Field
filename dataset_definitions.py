from __future__ import annotations

from os import path
from typing import *
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from shared import *

class DatasetDefinition:
    def __init__(self) -> None:
        self.wav_path: str = None
        self.is_f0_latent: bool = None
        # is f0 and amp latent

DAN   = './dataset/dan.wav'
YANHE = './dataset/yanhe.wav'
VOICE_SCALE = './dataset/voice_scale.wav'
MSS_0 = './dataset/URMP/01_Jupiter_vn_vc/AuMix_01_Jupiter_vn_vc.wav'

sample_page_lookup = {
    VOICE_SCALE: round(.25 * SR / PAGE_LEN * 2), 
}

dF = DatasetDefinition()
dF.wav_path = DAN
dF.is_f0_latent = False
danF0NotLatent = dF

dF = DatasetDefinition()
dF.wav_path = YANHE
dF.is_f0_latent = False
heF0NotLatent = dF

dF = DatasetDefinition()
dF.wav_path = DAN
dF.is_f0_latent = True
danF0IsLatent = dF

dF = DatasetDefinition()
dF.wav_path = YANHE
dF.is_f0_latent = True
heF0IsLatent = dF

dF = DatasetDefinition()
dF.wav_path = VOICE_SCALE
dF.is_f0_latent = True
voiceScaleF0IsLatent = dF

dF = DatasetDefinition()
dF.wav_path = MSS_0
dF.is_f0_latent = True
mss_0 = dF

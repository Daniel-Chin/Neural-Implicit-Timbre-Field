__all__ = [
    'HyperParams', 
]

from typing import *

import numpy as np
import torch
from torchWork import *

IntPair = Tuple[int, int]
IntOrPair = Union[int, IntPair]

class HyperParams(BaseHyperParams):
    def __init__(self) -> None:
        super().__init__()

        self.nickname: Optional[str] = None
        
        self.nif_width: int = None
        self.nif_depth: int = None
        self.n_vowel_dims: int = None

        self.nif_sees_f0: bool = None
        self.nif_sees_amp: bool = None
        self.nif_sees_vowel: bool = None
        self.nif_abs_out: bool = None
        self.nif_abs_confidence: bool = None
        self.nif_renorm_confidence: bool = None
        self.nif_fast_lr: float = None

        self.ground_truth_f0: bool = None

        self.batch_size: int = None
        # self.grad_clip: Optional[float] = None
        self.optim_name: str = None

        self.max_epoch: int = None

        self.experiment_globals: Dict = None

    def fillDefaults(self):
        '''
        This is necessary when we want to load old 
        experiments (with less hyper params) without 
        checking out old commits.  
        The default values should guarantee old behaviors.  
        '''
        pass

    def ready(self, experiment_globals):
        self.experiment_globals = experiment_globals

        self.OptimClass = {
            'adam': torch.optim.Adam, 
        }[self.optim_name]

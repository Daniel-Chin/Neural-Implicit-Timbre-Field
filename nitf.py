from itertools import chain

import torch
from torch import nn
from torchWork import DEVICE

from shared import *
from hyper_params import HyperParams
from dataset_definitions import DatasetDefinition
from dataset import MyDataset
from dredge import *

class NITF(nn.Module):
    def __init__(
        self, hParams: HyperParams, is_vocal: bool = True, 
    ) -> None:
        super().__init__()
        layers = []
        in_dim = 3  # freq, f0, amp
        if is_vocal:
            in_dim += hParams.n_vowel_dims
        dim = in_dim
        for _ in range(hParams.nif_depth):
            layers.append(nn.Linear(dim, hParams.nif_width))
            layers.append(nn.ReLU())
            dim = hParams.nif_width
        layers.append(nn.Linear(dim, 1))
        self.sequential = nn.Sequential(*layers)

        if is_vocal:
            datasetDef: DatasetDefinition = hParams.experiment_globals['datasetDef']
            dataset: MyDataset = hParams.experiment_globals['dataset']

            self.vowel_embs = torch.zeros(
                (dataset.n_pages, hParams.n_vowel_dims), 
                requires_grad=True, 
                device=DEVICE, 
            )
            self.register_buffer('saved_vowel_embs', self.vowel_embs)
            if datasetDef.is_f0_latent:
                self.dredge_freq = torch.ones(
                    (dataset.n_pages, ), 
                    device=DEVICE, 
                ).float() * 200 / FREQ_SCALE
                self.dredge_freq.requires_grad = True

                self.dredge_confidence = torch.zeros(
                    (dataset.n_pages, DREDGE_LEN), 
                    device=DEVICE, 
                ).float()
                self.dredge_confidence[:, DREDGE_RADIUS] = 1
                self.dredge_confidence.requires_grad = True

                self.amp_latent = torch.ones(
                    (dataset.n_pages, ), 
                    device=DEVICE, 
                ).float()
                self.amp_latent.requires_grad = True

                self.register_buffer(
                    'saved_dredge_freq', self.dredge_freq, 
                )
                self.register_buffer(
                    'saved_dredge_confidence', self.dredge_confidence, 
                )
                self.register_buffer(
                    'saved_amp_latent', self.amp_latent, 
                )
    
    def parameters(self):
        return chain(super().parameters(), self.buffers())
    
    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
    
    def simplifyDredge(self, optim: torch.optim.Optimizer):
        optim.state.clear()
        with torch.no_grad():
            old = self.dredge_confidence.clone()
            old_max, i = old.max(dim=1)
            where = old_max / old[:, DREDGE_RADIUS] >= 2
            i_where = i[where]
            self.dredge_confidence[where, :].fill_(0)
            self.dredge_confidence[where, DREDGE_RADIUS] = old[where, i_where]
            self.dredge_confidence[where, DREDGE_RADIUS * 2 - i_where] = old[where, DREDGE_RADIUS]
            self.dredge_confidence.grad[where, :] = 0
            self.dredge_freq[where] = self.dredge_freq[where] * DREDGE_MULT[i_where]
            self.dredge_freq.grad[where] = 0

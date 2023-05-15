from itertools import chain

import torch
from torch import nn

from hyper_params import HyperParams
from dataset_definitions import DatasetDefinition
from dataset import MyDataset

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
            )
            self.register_buffer('saved_vowel_embs', self.vowel_embs)
            if datasetDef.is_f0_latent:
                self.f0_latent = (torch.ones(
                    (dataset.n_pages, ), 
                    requires_grad=True, 
                ) * 300).detach()
                self.amp_latent = torch.ones(
                    (dataset.n_pages, ), 
                    requires_grad=True, 
                )
                self.register_buffer(
                    'saved_f0_latent', self.f0_latent, 
                )
                self.register_buffer(
                    'saved_amp_latent', self.amp_latent, 
                )
    
    def parameters(self):
        return chain(super().parameters(), self.buffers())
    
    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

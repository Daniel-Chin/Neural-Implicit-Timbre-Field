__all__ = [
    'DREDGE_RADIUS', 
    'DREDGE_LEN', 
    'DREDGE_MULT', 
    'regularizeDredge', 
]

import torch
from torchWork import DEVICE

RATIOS = sorted([2, 3, 3/2])
DREDGE_RADIUS = len(RATIOS)
DREDGE_LEN = DREDGE_RADIUS * 2 + 1
DREDGE_MULT = torch.ones((DREDGE_LEN, )).float().to(DEVICE)
for i in range(DREDGE_RADIUS):
    DREDGE_MULT[DREDGE_RADIUS - i - 1] = 1 / RATIOS[i]
    DREDGE_MULT[DREDGE_RADIUS + i + 1] = RATIOS[i]

def regularizeDredge(confidence: torch.Tensor):
    # confidence is (batch_size, DREDGE_LEN)
    dont_negative = confidence.clip(max=0).sum()
    # it's sum not mean because the to-step tensor (confidence)'s size scales with batch size. 
    positive = confidence.clip(min=0)
    positive_detached = positive.detach()
    unecessary_high_f0 = (
        positive * positive_detached.cumsum(dim=1)
    ).sum()
    return dont_negative + unecessary_high_f0

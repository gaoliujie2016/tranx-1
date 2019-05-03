import copy

import numpy as np
import torch
import torch.nn as nn


def clones(module, N):
    """Produce N identical layers."""

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."

    attn_shape = (1, size, size)
    subsq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsq_mask) == 0

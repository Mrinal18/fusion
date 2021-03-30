import abc

from typing import Optional

import torch.nn as nn
from tensor import Tensor


class ABaseLoss(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        pass

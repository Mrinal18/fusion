import abc

from typing import Optional

import torch.nn as nn
from tensor import Tensor

from fusion.model import ModelOutput


class ABaseLoss(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, preds: ModelOutput, target: Optional[Tensor] = None) -> Tensor:
        pass

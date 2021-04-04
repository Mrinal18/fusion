from typing import Optional

import torch.nn as nn
from torch import Tensor

from fusion.model.misc import ModelOutput

from . import ABaseLoss



class AE(ABaseLoss):
    def __init__(self, **kwargs):
        """

        kwargs:
        """
        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds: ModelOutput, target: Optional[Tensor] = None) -> Tensor:
        """

        preds:
        target:
        :return:
        """
        assert target is not None
        x_hat = preds.attrs['x_hat']
        x = preds.attrs['x']
        loss = self._loss(x_hat, x)
        return loss

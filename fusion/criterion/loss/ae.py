from typing import Optional

import torch.nn as nn
from tensor import Tensor

from . import ABaseLoss
from fusion.model import ModelOutput


class AE(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds: ModelOutput, target: Optional[Tensor] = None) -> Tensor:
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        x_hat = preds.attrs['x_hat']
        x = preds.attrs['x']
        loss = self._loss(x_hat, x)
        return loss

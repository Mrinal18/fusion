from typing import Optional

import torch.nn as nn
from tensor import Tensor

from . import ABaseLoss

class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(CrossEntropyLoss, self).__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        return self._loss(preds, target)


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        loss = self._loss(preds.squeeze(1), target.float())
        return loss

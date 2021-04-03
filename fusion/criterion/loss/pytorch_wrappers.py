from typing import Optional

import torch.nn as nn
from torch import Tensor

from . import ABaseLoss

class CustomCrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Cross Entropy Loss
        Args:
            :param kwargs: parameters of Cross Entropy Loss
        Return
            Class of Cross Entropy Loss
        """
        super().__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Forward method of class Cross Entropy Loss
        Args:
            :param preds: input tensor
            :param target: target tensor

        Return:
            Cross Entropy Loss between input and target tensor
        """
        return self._loss(preds, target)


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class MSE Loss
        Args:
            :param kwargs: parameters of MSE Loss
        Return
            Class of MSE Loss
        """

        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Forward method of class MSE Loss
        Args:
            :param preds: input tensor
            :param target: target tensor

        Return:
            MSE Loss between input and target tensor
        """
        assert target is not None
        return self._loss(preds.z[0], target)


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Binary Cross Entropy with
         logits loss
        Args:
            :param kwargs: parameters of Cross Entropy Loss
        Return
            Class of Binary Cross Entropy with logits loss
        """
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Forward method of class Binary Cross Entropy with
         logits loss
        Args:
            :param preds: input tensor
            :param target: target tensor

        Return:
             Class of Binary Cross Entropy with logits loss
             between input and target tensor
        """
        assert target is not None
        loss = self._loss(preds.squeeze(1), target.float())
        return loss

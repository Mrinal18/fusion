from typing import Optional

from fusion.model.misc import ModelOutput

import torch.nn as nn
from torch import Tensor

from . import ABaseLoss

class CustomCrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Cross Entropy Loss
        Args:
            kwargs: parameters of Cross Entropy Loss
        Return
            Class of Cross Entropy Loss
        """
        super().__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds: ModelOutput, target: Optional[Tensor] = None) -> Tensor:
        """
        Forward method of class Cross Entropy Loss
        Args:
            preds: input model's output
            target: target tensor

        Return:
            Cross Entropy Loss between input and target tensor
        """
        ret_loss = None
        raw_losses = {}
        for source_id, z in preds.z.items():
            loss = self._loss(z, target)
            ret_loss = ret_loss + loss if ret_loss is not None else loss
            raw_losses[f"CE{source_id}"] = loss
        return ret_loss, raw_losses


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class MSE Loss
        Args:
            kwargs: parameters of MSE Loss
        Return
            Class of MSE Loss
        """

        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Forward method of class MSE Loss
        Args:
            preds: input tensor
            target: target tensor

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
            kwargs: parameters of Cross Entropy Loss
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
            preds: input tensor
            target: target tensor

        Return:
             Class of Binary Cross Entropy with logits loss
             between input and target tensor
        """
        assert target is not None
        ret_loss = None
        raw_losses = {}
        for source_id, z in preds.z.items():
            loss = self._loss(z.squeeze(1), target)
            ret_loss = ret_loss + loss if ret_loss is not None else loss
            raw_losses[f"CE{source_id}"] = loss
        return ret_loss, raw_losses

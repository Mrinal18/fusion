from . import ABaseLoss

import torch.nn as nn


class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, target=None):
        assert target is not None
        return self._loss(preds, target)


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds, target=None):
        assert target is not None
        loss = self._loss(preds, target)
        return loss


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds, target=None):
        assert target is not None
        loss = self._loss(preds.squeeze(1), target.float())
        return loss

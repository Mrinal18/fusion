from . import ABaseLoss

import torch.nn as nn


class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input, target):
        return self._loss(input, target)


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        loss = self._loss(input, target)
        return loss


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, input, target):
        loss = self._loss(input.squeeze(1), target.float())
        return loss

from . import ABaseLoss

import torch.nn as nn


class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(CrossEntropyLoss, self).__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, target=None):
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        return self._loss(preds.z[0], target)


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(BCEWithLogitsLoss, self).__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds, target=None):
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        loss = self._loss(preds.squeeze(1), target.float())
        return loss

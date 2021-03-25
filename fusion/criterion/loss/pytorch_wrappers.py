from . import ABaseLoss

import torch.nn as nn


class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input, target):
        """

        :param input:
        :param target:
        :return:
        """
        return self.loss(input, target)


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        """

        :param input:
        :param target:
        :return:
        """
        loss = self.loss(input, target)
        return loss


class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, input, target):
        """

        :param input:
        :param target:
        :return:
        """
        loss = self.loss(input.squeeze(1), target.float())
        return loss

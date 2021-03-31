from . import ABaseLoss

import torch.nn as nn


class CrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Cross Entropy Loss
        Args:
            :param kwargs: parameters of Cross Entropy Loss
        Return
            Class of Cross Entropy Loss
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input, target):
        """
        Forward method of class Cross Entropy Loss
        Args:
            :param input: input tensor
            :param target: target tensor

        Return:
            Cross Entropy Loss between input and target tensor
        """
        return self.loss(input, target)


class MSELoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class MSE Loss
        Args:
            :param kwargs: parameters of MSE Loss
        Return
            Class of MSE Loss
        """

        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        """
        Forward method of class MSE Loss
        Args:
            :param input: input tensor
            :param target: target tensor

        Return:
            MSE Loss between input and target tensor
        """
        loss = self.loss(input, target)
        return loss


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
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, input, target):
        """
        Forward method of class Binary Cross Entropy with
         logits loss
        Args:
            :param input: input tensor
            :param target: target tensor

        Return:
             Class of Binary Cross Entropy with logits loss
             between input and target tensor
        """
        loss = self.loss(input.squeeze(1), target.float())
        return loss

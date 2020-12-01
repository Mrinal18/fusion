import abc
import torch.nn as nn


class ABaseLoss(abc.ABC, nn.Module):
    _loss = None

    @abc.abstractmethod
    def __init__(self):
        super(ABaseLoss, self).__init__()

    def forward(self, input, target):
        pass

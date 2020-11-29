import abc
import torch.nn as nn


class ABaseLoss(abc.ABC, nn.Module):
    _loss = None

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    def forward(self, input, target):
        pass

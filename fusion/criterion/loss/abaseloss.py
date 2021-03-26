import abc
import torch.nn as nn


class ABaseLoss(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super(ABaseLoss, self).__init__()

    def forward(self, preds, target=None):
        pass

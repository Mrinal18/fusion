from . import ABaseLoss
import torch.nn as nn


class AE(ABaseLoss):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(AE, self).__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(self, preds, target=None):
        """

        :param preds:
        :param target:
        :return:
        """
        assert target is not None
        x_hat = preds.attrs['x_hat']
        x = preds.attrs['x']
        loss = self._loss(x_hat, x)
        return loss
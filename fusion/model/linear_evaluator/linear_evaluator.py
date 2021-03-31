import torch.nn as nn
from torch import Tensor

from fusion.architecture.base_block import Flatten


class LinearEvaluator(nn.Module):
    def __init__(self, encoder, num_classes: int, dim_l: int, source_id: int):
        """

        :param encoder:
        :param num_classes:
        :param dim_l:
        :param source_id:
        """
        super().__init__()
        self._encoder = encoder
        self._encoder.eval()
        self._flatten = Flatten()
        self._linear = nn.Linear(dim_l, num_classes)
        self._source_id = source_id

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x:
        :return:
        """
        x = x[self._source_id]
        x = self._encoder(x)[0]
        x = x.detach()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self._flatten(x)
        x = self._linear(x)
        return x

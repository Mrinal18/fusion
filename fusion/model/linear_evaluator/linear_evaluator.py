import torch.nn as nn
from torch import Tensor

from fusion.architecture.base_block import Flatten
from fusion.model.misc import ModelOutput


class LinearEvaluator(nn.Module):
    def __init__(self, encoder, num_classes: int, dim_l: int, source_id: int):
        """

        encoder:
        num_classes:
        dim_l:
        source_id:
        """
        super().__init__()
        self._encoder = encoder
        self._encoder.eval()
        self._flatten = Flatten()
        self._linear = nn.Linear(dim_l, num_classes)
        self._source_id = source_id

    def forward(self, x: Tensor) -> ModelOutput:
        """

        x:
        :return:
        """
        x = x[self._source_id]
        x = self._encoder(x)[0]
        x = x.detach()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self._flatten(x)
        x = self._linear(x)
        return ModelOutput(z={0: x}, attrs={})

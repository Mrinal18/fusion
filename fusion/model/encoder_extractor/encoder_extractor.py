import torch.nn as nn
from torch import Tensor

from fusion.model.misc import ModelOutput


class EncoderExtractor(nn.Module):
    def __init__(self, encoder, source_id: int):
        """

        encoder:
        num_classes:
        dim_l:
        source_id:
        """
        super().__init__()
        self._encoder = encoder
        self._encoder.eval()
        self._source_id = source_id

    def forward(self, x: Tensor) -> ModelOutput:
        """

        x:
        :return:
        """
        source_id = self._source_id if len(x) > 1 else 0
        x = x[source_id]
        x = self._encoder(x)[0]
        x = x.detach()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return ModelOutput(z={self._source_id: x}, attrs={})

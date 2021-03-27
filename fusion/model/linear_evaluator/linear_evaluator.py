from fusion.architecture.base_block import Flatten
import torch.nn as nn


class LinearEvaluator(nn.Module):
    def __init__(self, encoder, num_classes, dim_l, source_id):
        """

        :param encoder:
        :param num_classes:
        :param dim_l:
        :param source_id:
        """
        super(LinearEvaluator, self).__init__()
        self._encoder = encoder
        self._encoder.eval()
        self._flatten = Flatten()
        self._linear = nn.Linear(dim_l, num_classes)
        self._source_id = source_id

    def forward(self, x):
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

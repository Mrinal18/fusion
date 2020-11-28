from fusion.model import AUniSourceModel
import torch.nn as nn


class Supervised(AUniSourceModel):
    def __init__(
        self,
        dim_l,
        num_classes,
        architecture,
        architecture_params
    ):
        self._encoder = architecture(**architecture_params)
        self._linear = nn.Linear(dim_l, num_classes)

    def forward(self, x):
        x, _ = self._encoder(x)
        x = self.linear(x)
        return x

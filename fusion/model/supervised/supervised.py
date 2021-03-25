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
        """
        Initialization of supervise model
        :param dim_l: output dimension of encoder
        :param num_classes: number of classes
        :param architecture: type of architecture
        :param architecture_params: parameters of architecture
        """
        super(Supervised, self).__init__(architecture, architecture_params)
        self._linear = nn.Linear(dim_l, num_classes)

    def forward(self, x):
        """
        Forward method of supervised models
        :param x: input tensor
        :return:
        result of forward propagation
        """
        x, _ = self._encoder(x[0])
        x = self._linear(x)
        return x

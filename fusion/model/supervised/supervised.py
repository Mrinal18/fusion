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
        Args:
            :param dim_l: output dimension of encoder
            :param num_classes: number of classes
            :param architecture: type of architecture
            :param architecture_params: parameters of architecture
        Return:
            Supervise model

        """
        super(Supervised, self).__init__(architecture, architecture_params)
        self._linear = nn.Linear(dim_l, num_classes)

    def forward(self, x):
        """
        Forward method of supervised models
        Args:
            :param x: input tensor
        Return:
            result of forward propagation
        """
        x, _ = self._encoder(x[0])
        x = self._linear(x)
        return x

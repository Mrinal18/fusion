from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput
import torch.nn as nn


class Supervised(ABaseModel):
    def __init__(
        self,
        dim_l,
        num_classes,
        sources,
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
        super(Supervised, self).__init__(sources, architecture, architecture_params)
        assert len(sources) == 1
        self._sources = sources
        self._linear = nn.Linear(dim_l, num_classes)

    def forward(self, x):
        """
        Forward method of supervised models
        :param x: input tensor
        :return:
        result of forward propagation
        """
        assert len(x) == 1
        x = self._source_forward(self._sources[0], x)
        return ModelOutput(z={0: x}, attrs={})

    def _source_forward(self, source_id, x):
        x, _ = self._encoder[str(source_id)](x[0])
        x = self._linear(x)
        return x

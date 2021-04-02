from typing import Any, Dict, List

import torch.nn as nn
from torch import Tensor

from fusion.model import ABaseModel


class Supervised(ABaseModel):
    def __init__(
        self,
        dim_l: int,
        num_classes: int,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any]
    ):
        """
        Initialization of supervise model
        :param dim_l: output dimension of encoder
        :param num_classes: number of classes
        :param architecture: type of architecture
        :param architecture_params: parameters of architecture
        """
        super().__init__(sources, architecture, architecture_params)
        assert len(sources) == 1
        self._sources = sources
        self._linear = nn.Linear(dim_l, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method of supervised models
        :param x: input tensor
        :return:
        result of forward propagation
        """
        assert len(x) == 1
        x = self._source_forward(self._sources[0], x)
        return x

    def _source_forward(self, source_id: int, x: Tensor) -> Tensor:
        x, _ = self._encoder[str(source_id)](x[0])
        x = self._linear(x)
        return x

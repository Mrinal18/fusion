from typing import Any, Dict, List, Tuple

from torch import Tensor

from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class AE(ABaseModel):
    def __init__(self, sources: List[int], architecture: str, architecture_params: Dict[str, Any]):
        """
        Initialization class of autoencoder model

        Args:
            :param sources:
            :param architecture: type of architecture
            :param architecture_params: parameters of architecture

        Return:
            Autoencoder model
        """
        super().__init__(sources, architecture, architecture_params)

    def _source_forward(self, source_id: int, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """

        :param source_id:
        :param x:
        :return:
        """
        return self._encoder[source_id](x[int(source_id)])

    def forward(self, x: Tensor) -> ModelOutput:
        """
        Forward method of autoencoder model

        Args:

            :param x: input tensor
        Return:
            Result of forward method of autoencoder model

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs['x'] = {}
        ret.attrs['x_hat'] = {}
        for source_id, _ in self._encoder.items():
            ret.attrs['x'] = x[int(source_id)]
            z, x_hat = self._source_forward(source_id, x)
            ret.z[source_id] = z
            ret.attrs['x_hat'] = x_hat
        return ret

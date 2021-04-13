from typing import Any, Dict, List, Tuple

from torch import Tensor

from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class AE(ABaseModel):
    def __init__(self, sources: List[int], architecture: str, architecture_params: Dict[str, Any]):
        """
        Initialization class of autoencoder model

        Args:
            sources:
            architecture: type of architecture
            architecture_params: parameters of architecture

        Return:
            Autoencoder model
        """
        super().__init__(sources, architecture, architecture_params)

    def _source_forward(self, source_id: int, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """

        source_id:
        x:
        :return:
        """
        source_id_int = int(source_id)
        source_id_s = str(source_id)
        if len(self._sources) == 1:
            source_id_int = 0
        return self._encoder[source_id_s](x[source_id_int])

    def forward(self, x: ModelOutput) -> ModelOutput:
        """
        Forward method of autoencoder model

        Args:

            x: input tensor
        Return:
            Result of forward method of autoencoder model

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs['x'] = {}
        ret.attrs['x_hat'] = {}
        ret.attrs['latents'] = {}
        for source_id, _ in self._encoder.items():
            source_id_int = int(source_id)
            if len(self._sources) == 1:
                source_id_int = 0
            z, x_hat = self._source_forward(source_id, x)
            ret.z[int(source_id)] = z
            ret.attrs['latents'][int(source_id)] = {1: z}
            ret.attrs['x'][int(source_id)] = x[source_id_int]
            ret.attrs['x_hat'][int(source_id)] = x_hat
        return ret

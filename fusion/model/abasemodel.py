import abc
import copy
from fusion.architecture import architecture_provider
import torch.nn as nn


class ABaseModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self, sources, architecture, architecture_params):
        """

         :param sources:
         :param architecture:
         :param architecture_params:
         """
        super(ABaseModel, self).__init__()
        self._sources = sources
        self._encoder = nn.ModuleDict({})
        for i, source_id in enumerate(self._sources):
            new_architecture_params = copy.deepcopy(architecture_params)
            new_architecture_params['dim_in'] = architecture_params['dim_in'][i]
            self._encoder[str(source_id)] = architecture_provider.get(
                architecture, **new_architecture_params)

    @abc.abstractmethod
    def _source_forward(self, source_id, x):
        pass

    def get_encoder(self, source_id=0):
        assert source_id in self._encoder.keys()
        return self._encoder[str(source_id)]

    def get_encoder_list(self):
        return self._encoder


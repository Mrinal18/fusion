import abc
import copy
from fusion.architecture import architecture_provider
import torch.nn as nn


class BaseModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def get_encoder(self, source_id=0):
        pass


class AMultiSourceModel(BaseModel):

    @abc.abstractmethod
    def __init__(self, sources, architecture, architecture_params):
        super(AMultiSourceModel, self).__init__()
        architecture_params = dict(**architecture_params)
        self._views = sources
        self._model = nn.ModuleDict({})
        for i, view in enumerate(self._views):
            new_architecture_params = copy.deepcopy(architecture_params)
            new_architecture_params['dim_in'] = architecture_params['dim_in'][i]
            self._model[view] = architecture_provider.get(
                architecture, **architecture_params)

    @abc.abstractmethod
    def _source_forward(self, source_id, x):
        pass

    def get_encoder(self, source_id=0):
        assert source_id in self._model.keys()
        return self._model[source_id]


class AUniSourceModel(BaseModel):
    @abc.abstractmethod
    def __init__(self, architecture, architecture_params):
        super(AUniSourceModel, self).__init__()
        architecture_params = dict(**architecture_params)
        architecture_params['dim_in'] = architecture_params['dim_in'][0]
        self._encoder = architecture_provider.get(
            architecture, **architecture_params)

    def get_encoder(self, source_id=0):
        del source_id
        return self._encoder


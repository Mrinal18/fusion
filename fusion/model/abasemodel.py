import abc
from abc import abstractmethod
import torch.nn as nn


class BaseModel(abc.ABC, nn.Module):
    _encoder = None
    @abc.abstractmethod
    def __init__(
        self,
        architecture_class,
        architecture_params
    ):
        pass

    @abc.abstractmethod
    def get_encoder(self, source_id=0):
        pass


class AMultiSourceModel(BaseModel):
    _encoder = {}

    @abc.abstractmethod
    def _source_forward(source_id):
        pass

    @abc.abstractmethod
    def get_encoder(self, source_id=0):
        assert source_id in self._encoder.keys()
        return self._encoder[source_id]


class AUniSourceModel(BaseModel):

    @abc.abstractmethod
    def get_encoder(self, source_id=0):
        del source_id
        return self._encoder


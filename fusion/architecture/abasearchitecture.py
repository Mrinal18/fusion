import abc
from typing import Optional, Type

import torch.nn as nn


TActivation = Type[nn.Module]
TDropout = Type[nn.modules.dropout._DropoutNd]
TConv = Type[nn.modules.conv._ConvNd]
TNorm = Type[nn.modules.batchnorm._BatchNorm]


class ABaseArchitecture(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(
        self,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = None,
        dp_layer_class: Optional[TDropout] = None,
        activation_class: Optional[TActivation] = None,
        weights_initialization_type: Optional[str] = None,
    ):
        """

        conv_layer_class:
        norm_layer_class:
        dp_layer_class:
        activation_class:
        weights_initialization_type:
        """
        super().__init__()
        self._layers: Optional[nn.ModuleList] = None
        self._conv_layer_class = conv_layer_class
        self._norm_layer_class = norm_layer_class
        self._dp_layer_class = dp_layer_class
        self._activation_class = activation_class
        self._weights_initialization_type = weights_initialization_type

    @abc.abstractmethod
    def init_weights(self):
        """Weight initialization"""
        pass

    def get_layers(self):
        """
        Get layers
        :return:
        Layers

        """
        return self._layers

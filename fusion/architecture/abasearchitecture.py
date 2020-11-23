import abc
from abc import abstractmethod
import torch.nn as nn


class ABaseArchitecture(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(
        self,
        conv_layer_class=nn.Conv2d,
        norm_layer_class=None,
        dp_layer_class=None,
        activation_class=None,
        weights_initlization_type=None
    ):
        super(ABaseArchitecture, self).__init__()
        self._conv_layer_class = conv_layer_class
        self._norm_layer_class = norm_layer_class
        self._dp_layer_class = dp_layer_class
        self._activation_class = activation_class
        self._weights_initlization_type = weights_initlization_type

    @abc.abstractmethod
    def _init_weights():
        """Weight initilization
        """
        pass

import abc
import torch.nn as nn


class ABaseArchitecture(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(
        self,
        conv_layer_class=nn.Conv2d,
        norm_layer_class=None,
        dp_layer_class=None,
        activation_class=None,
        weights_initialization_type=None
    ):
        """

        :param conv_layer_class:
        :param norm_layer_class:
        :param dp_layer_class:
        :param activation_class:
        :param weights_initialization_type:
        """
        super(ABaseArchitecture, self).__init__()
        self._layers = None
        self._conv_layer_class = conv_layer_class
        self._norm_layer_class = norm_layer_class
        self._dp_layer_class = dp_layer_class
        self._activation_class = activation_class
        self._weights_initialization_type = weights_initialization_type

    @abc.abstractmethod
    def init_weights(self):
        """Weight initialization
        """
        pass

    def get_layers(self):
        """
        Get layers
        :return:
        Layers

        """
        return self._layers

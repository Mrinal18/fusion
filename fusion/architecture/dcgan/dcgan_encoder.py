from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm
from fusion.architecture.base_block import BaseConvLayer, Flatten


class DcganEncoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_l: int,
        dim_cls=None,
        input_size: int = 32,
        conv_layer_class: TConv = nn.Conv2d,
        norm_layer_class: TNorm = nn.BatchNorm2d,
        activation_class: TActivation = nn.LeakyReLU,
        weights_initialization_type: str = 'xavier_uniform',
    ):
        """
        The DCGAN Encoder class
        Args:
            :param dim_in: The number of input channels
            :param dim_h: The number of feature channels for the first convolutional layer, the number of feature channels double with each next convolutional layer
            :param dim_l: The number of latent dimensions
            :param dim_cls: A list of scalars, where each number should correspond to the output width for one of the convolutional layers. 
                             The information between latent variable z and the convolutional feature maps width widths in dim_cls are maximized.
                             If dim_cls=None, the information between z and none of the convolutional feature maps is maximized, default=None
            :param input_size: The input width and height of the image, default=32
            :param conv_layer_class: The type of convolutional layer to use, default=nn.Conv2d
            :param norm_layer_class: he type of normalization layer to use, default=nn.BatchNorm2d
            :param activation_class: The type of non-linear activation function to use, default=nn.LeakyReLU
            :param weights_initialization_type: The weight initialization type to use, default='xavier_uniform'
        """
        super().__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._dim_in = dim_in
        self._dim_h = dim_h
        self._dim_l = dim_l
        self._dim_cls = dim_cls
        self._input_size = input_size
        self._flatten = Flatten()
        self._layers: nn.ModuleList

        self._construct()
        self.init_weights()

    def _construct(self):
        self._layers = nn.ModuleList([
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': self._dim_in, 'out_channels': self._dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                activation_class=self._activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': self._dim_h, 'out_channels': 2 * self._dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=self._norm_layer_class, norm_layer_args={
                    'num_features': 2 * self._dim_h
                },
                activation_class=self._activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': 2 * self._dim_h, 'out_channels': 4 * self._dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=self._norm_layer_class, norm_layer_args={
                    'num_features': 4 * self._dim_h
                },
                activation_class=self._activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
        ])
        if self._input_size == 64:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels': 4 * self._dim_h, 'out_channels': 8 * self._dim_h,
                        'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                    },
                    norm_layer_class=self._norm_layer_class, norm_layer_args={
                        'num_features': 8 * self._dim_h
                    },
                    activation_class=self._activation_class, activation_args={
                        'negative_slope': 0.2, 'inplace': True
                    }
                )
            )
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels': 8 * self._dim_h, 'out_channels': self._dim_l,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                ),
            )
        elif self._input_size == 32:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels': 4 * self._dim_h, 'out_channels': self._dim_l,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                )
            )
        else:
            raise NotImplementedError("DCGAN only supports input square images ' + \
                'with size 32, 64 in current implementation.")


    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        The DCGAN encoder forward method
        Args:
            :param x: The input tensor
        Returns:
            z: The latent variable
            latents: The convolutional feature maps, with widths specified by self._dim_cls
        """
        latents = None
        if self._dim_cls is not None:
            latents = {}
        for layer in self._layers:
            x, conv_latent = layer(x)
            # Add conv latent
            if self._dim_cls is not None:
                if conv_latent.size()[-1] in self._dim_cls:
                    latents[conv_latent.size()[-1]] = conv_latent
        # Adds latent
        if self._dim_cls is not None:
            latents[1] = x
        # Flatten to get representation
        z = self._flatten(x)
        return z, latents

    def init_weights(self):
        """
        Weight initialization method
        Returns:
            DcganEncoder with initialized weights

        """
        for layer in self._layers:
            layer.init_weights(gain_type='leaky_relu')

from typing import Dict, Iterable, Tuple, Type

import torch.nn as nn
from torch import Tensor

from fusion.architecture import ABaseArchitecture
from fusion.architecture.base_block import BaseConvLayer, Unflatten


class DcganDecoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_l: int,
        dim_cls=None,
        input_size: int = 32,
        input_dim: int = 2,
        conv_layer_class: Type[nn.modules.conv._ConvNd] = nn.ConvTranspose2d,
        norm_layer_class: Type[nn.modules.batchnorm._BatchNorm] = nn.BatchNorm2d,
        activation_class: Type[nn.Module] = nn.ReLU,
        weights_initialization_type: str = 'xavier_uniform',
    ):
        """

        :param dim_in:
        :param dim_h:
        :param dim_l:
        :param dim_cls:
        :param input_size:
        :param input_dim:
        :param conv_layer_class:
        :param norm_layer_class:
        :param activation_class:
        :param weights_initialization_type:
        """
        super().__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type
        )
        self._dim_in = dim_in
        self._dim_h = dim_h
        self._dim_l = dim_l
        self._dim_cls = dim_cls
        self._input_size = input_size
        self._unflatten = Unflatten(input_dim=input_dim)
        self._layers: Iterable[nn.Module] = nn.ModuleList([])
        self._construct()

    def _construct(self):
        if self._input_size == 64:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels':  self._dim_l, 'out_channels': 8 *  self._dim_h,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                    norm_layer_class= self._norm_layer_class, norm_layer_args={
                        'num_features': 8 * self._dim_h
                    },
                    activation_class= self._activation_class, activation_args={
                        'inplace': True
                    }
                )
            )
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels': 8 * self._dim_h, 'out_channels': 4 * self._dim_h,
                        'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                    },
                    norm_layer_class=self._norm_layer_class, norm_layer_args={
                        'num_features': 4 * self._dim_h
                    },
                    activation_class=self._activation_class, activation_args={
                        'inplace': True
                    }
                )
            )
        elif self._input_size == 32:
            self._layers.append(
                BaseConvLayer(
                    self._conv_layer_class, {
                        'in_channels': self._dim_l, 'out_channels': 4 * self._dim_h,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                    norm_layer_class=self._norm_layer_class, norm_layer_args={
                        'num_features': 4 * self._dim_h
                    },
                    activation_class=self._activation_class, activation_args={
                        'inplace': True
                    }
                )
            )
        else:
            raise NotImplementedError("DCGAN only supports input square images ' + \
                'with size 32, 64 in current implementation.")

        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': 4 * self._dim_h, 'out_channels': 2 * self._dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=self._norm_layer_class, norm_layer_args={
                    'num_features': 2 * self._dim_h
                },
                activation_class=self._activation_class, activation_args={
                    'inplace': True
                }
            )
        )
        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': 2 * self._dim_h, 'out_channels': self._dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=self._norm_layer_class, norm_layer_args={
                    'num_features': self._dim_h
                },
                activation_class=self._activation_class, activation_args={
                    'inplace': True
                }
            )
        )
        self._layers.append(
            BaseConvLayer(
                self._conv_layer_class, {
                    'in_channels': self._dim_h, 'out_channels': self._dim_in,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                activation_class=nn.Tanh
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        x_hat = self._unflatten(x)
        latents = None
        # Adds latent
        if self._dim_cls is not None:
            latents = {}
            latents[1] = x_hat
        for layer in self._layers:
            x_hat, conv_latent = layer(x_hat)
            # Add conv latent
            if self._dim_cls is not None:
                if conv_latent.size()[-1] in self._dim_cls:
                    latents[conv_latent.size()[-1]] = conv_latent
        return x_hat, latents

    def init_weights(self):
        for layer in self._layers:
            layer.init_weights()

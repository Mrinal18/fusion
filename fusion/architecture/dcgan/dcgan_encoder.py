from fusion.architecture import ABaseArchitecture
from fusion.architecture.base_block import BaseConvLayer, Flatten
import torch.nn as nn


class DcganEncoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_l,
        dim_cls,
        input_size=32,
        conv_layer_class=nn.Conv2d,
        norm_layer_class=nn.BatchNorm2d,
        activation_class=nn.LeakyReLU,
        weights_initialization_type='xavier_uniform',
    ):
        super(DcganEncoder, self).__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type,
        )
        self._dim_cls = dim_cls
        self._flatten = Flatten()
        self._layers = nn.ModuleList([
            BaseConvLayer(
                conv_layer_class, {
                    'in_channels': dim_in, 'out_channels': dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                activation_class=activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
            BaseConvLayer(
                conv_layer_class, {
                    'in_channels': dim_h, 'out_channels': 2 * dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=norm_layer_class, norm_layer_args={
                    'num_features': 2 * dim_h
                },
                activation_class=activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
            BaseConvLayer(
                conv_layer_class, {
                    'in_channels': 2 * dim_h, 'out_channels': 4 * dim_h,
                    'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                },
                norm_layer_class=norm_layer_class, norm_layer_args={
                    'num_features': 4 * dim_h
                },
                activation_class=activation_class, activation_args={
                    'negative_slope': 0.2, 'inplace': True
                }
            ),
        ])
        if input_size == 64:
            self._layers.append(
                BaseConvLayer(
                    conv_layer_class, {
                        'in_channels': 4 * dim_h, 'out_channels': 8 * dim_h,
                        'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False
                    },
                    norm_layer_class=norm_layer_class, norm_layer_args={
                        'num_features': 8 * dim_h
                    },
                    activation_class=activation_class, activation_args={
                        'negative_slope': 0.2, 'inplace': True
                    }
                ),
                BaseConvLayer(
                    conv_layer_class, {
                        'in_channels': 8 * dim_h, 'out_channels': dim_l,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                ),
            )
        elif input_size == 32:
            self._layers.append(
                BaseConvLayer(
                    conv_layer_class, {
                        'in_channels': 4 * dim_h, 'out_channels': dim_l,
                        'kernel_size': 4, 'stride': 2, 'padding': 0, 'bias': False
                    },
                )
            )
        else:
            raise NotImplementedError("DCGAN only supports input square images ' + \
                'with size 32, 64 in current implementation.")
        self.init_weights()

    def forward(self, x):
        latents = {}
        for layer in self._layers:
            x, conv_latent = layer(x)
            # Add conv latent
            if conv_latent.size()[-1] in self._dim_cls:
                latents[conv_latent.size()[-1]] = conv_latent
        # Adds latent
        latents[1] = x
        # Flatten to get representation
        z = self._flatten(x)
        return z, latents

    def init_weights(self):
        for layer in self._layers:
            layer.init_weights()

from fusion.architecture.abasearchitecture import ABaseArchitecture
import torch.nn as nn


class BaseConvLayer(ABaseArchitecture):
    def __init__(
        self,
        conv_layer_class,
        conv_layer_args,
        norm_layer_class=None,
        norm_layer_args={},
        dp_layer_class=None,
        dp_layer_args={},
        activation_class=None,
        activation_args={},
        weights_initlization_type='xavier_uniform'
    ):
        super(BaseConvLayer, self).__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            dp_layer_class=dp_layer_class,
            activation_class=activation_class,
            weights_initlization_type=weights_initlization_type
        )
        self._layer = nn.ModuleList()
        self._layer.append(
            self._conv_layer_class(**conv_layer_args))
        if self._norm_layer_class:
            self._layer.append(
                self._norm_layer_class(**norm_layer_args)
            )
        if self._dp_layer_class:
            self._layer.append(
                self._dp_layer_class(**dp_layer_args)
            )
        if self._activation_class:
            self._layer.append(
                self._activation_class(**activation_args)
            )
        self._init_weights()

    def forward(self, x):
        x = self._layer[0]
        conv_latent = x
        for layer in self._layer[1:]:
            x = layer(x)
        return (x, conv_latent)

    def _init_weights(self):
        if self._weights_initlization_type == 'xavier_uniform':
            nn.init.xavier_uniform_(
                self._layer[0].weight, gain=nn.init.calculate_gain("relu")
            )
            if not isinstance(self._layer[0].bias, type(None)):
                nn.init.constant_(self._layer[0].bias, 0)
        else:
            raise NotImplementedError

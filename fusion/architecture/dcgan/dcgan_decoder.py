from fusion.architecture import ABaseArchitecture
from fusion.architecture.base_block import BaseConvLayer, Unflatten
import torch.nn as nn


class DcganDecoer(ABaseArchitecture):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_l,
        dim_cls=None,
        input_size=32,
        input_dim=2,
        conv_layer_class=nn.ConvTranspose2d,
        norm_layer_class=nn.BatchNorm2d,
        activation_class=nn.LeakyReLU,
        weights_initialization_type='xavier_uniform',
    ):
        super(DcganDecoer, self).__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type
        )
        self._dim_cls = dim_cls
        self._unflatten = Unflatten(input_dim=input_dim)
        self._layers = nn.ModuleList([])
        if input_size == 64:


    def forward(self, x):
        pass

    def init_weights(self):
        pass

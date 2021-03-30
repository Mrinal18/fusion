from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .dcgan_encoder import DcganEncoder
from .dcgan_decoder import DcganDecoder
from fusion.architecture import ABaseArchitecture
from fusion.architecture.abasearchitecture import TActivation, TConv, TNorm


class DcganAutoEncoder(ABaseArchitecture):
    def __init__(self,
         dim_in: int,
         dim_h: int,
         dim_l: int,
         dim_cls=None,
         input_size: int = 32,
         input_dim: int = 2,
         conv_layer_class: TConv = nn.Conv2d,
         conv_t_layer_class: TConv = nn.ConvTranspose2d,
         norm_layer_class: TNorm = nn.BatchNorm2d,
         activation_class: TActivation = nn.LeakyReLU,
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
        :param conv_t_layer_class:
        :param norm_layer_class:
        :param activation_class:
        :param weights_initialization_type:
        """
        super().__init__()
        self._encoder = DcganEncoder(
            dim_in, dim_h, dim_l, dim_cls=dim_cls,
            input_size=input_size, conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class, activation_class=activation_class,
            weights_initialization_type=weights_initialization_type
        )
        self._decoder = DcganDecoder(
            dim_in, dim_h, dim_l, dim_cls=dim_cls,
            input_size=input_size, input_dim=input_dim,
            conv_layer_class=conv_t_layer_class,
            norm_layer_class=norm_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type
        )

    def forward(self, x: Tensor) -> Tuple[Tuple, Tuple]:
        z, _ = self._encoder(x)
        x_hat, _ = self._decoder(z)
        return z, x_hat

    def init_weights(self):
        self._encoder.init_weights()
        self._decoder.init_weights()

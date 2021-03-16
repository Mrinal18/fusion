from .dcgan_encoder import DcganEncoder
from .dcgan_decoder import DcganDecoder
from fusion.architecture import ABaseArchitecture
import torch.nn as nn


class DcganAutoEncoder(ABaseArchitecture):
    def __init__(self,
         dim_in,
         dim_h,
         dim_l,
         dim_cls=None,
         input_size=32,
         input_dim=2,
         conv_layer_class=nn.Conv2d,
         conv_t_layer_class=nn.ConvTranspose2d,
         norm_layer_class=nn.BatchNorm2d,
         activation_class=nn.LeakyReLU,
         weights_initialization_type='xavier_uniform',
     ):
        super(DcganAutoEncoder, self).__init__()
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

    def forward(self, x):
        z, _ = self._encoder(x)
        x_hat, _ = self._decoder(z)
        return z, x_hat

    def init_weights(self):
        self._encoder.init_weights()
        self._decoder.init_weights()

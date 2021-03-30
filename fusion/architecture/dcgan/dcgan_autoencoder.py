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
        """
        Class of DCGAN autoencoder
        Args:
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

        Returns:
        Class of DCGAN autoencoder model


        """
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
        """
        Forward method of DCGAN autoencoder model
        Args:
            :param x:  input tensor
        Returns:
            z:
            x_hat:

        """
        z, _ = self._encoder(x)
        x_hat, _ = self._decoder(z)
        return z, x_hat

    def init_weights(self):
        """
        Method for initialization weights
        Return:
            Autoencoder with initialization weights

        """
        self._encoder.init_weights()
        self._decoder.init_weights()

from fusion.architecture import ABaseArchitecture
from fusion.architecture.dcgan import DcganEncoder
import torch
import torch.nn as nn


class VAEEncoder(ABaseArchitecture):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_l,
        dim_cls=None,
        input_size=32,
        conv_layer_class=nn.Conv2d,
        norm_layer_class=nn.BatchNorm2d,
        activation_class=nn.LeakyReLU,
        weights_initialization_type='xavier_uniform',
        use_last_layer=False
    ):
        super(VAEEncoder, self).__init__(
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

        self._dcgan_encoder = DcganEncoder(dim_in=dim_in, dim_h=dim_h,
                                           dim_l=dim_l, dim_cls=dim_cls,
                                           input_size=input_size,
                                           conv_layer_class=conv_layer_class,
                                           norm_layer_class=norm_layer_class,
                                           activation_class=activation_class,
                                           weights_initialization_type=weights_initialization_type,
                                           use_last_layer=use_last_layer)


        self._feature_map_size = self._test_feature_map_size()
        self._construct()

    def _test_feature_map_size(self):
        if self._conv_layer_class is nn.Conv1d:
            # Instantiate a 1-Dimensional tensor
            x = torch.rand(1,
                           self._dim_in,
                           self._input_size,
                           requires_grad=False)
        elif self._conv_layer_class is nn.Conv2d:
            # Instantiate a 2-Dimensional tensor
            x = torch.rand(1,
                           self._dim_in,
                           self._input_size,
                           self._input_size,
                           requires_grad=False)
        elif self._conv_layer_class is nn.Conv3d:
            # Instantiate a 3-Dimensional tensor
            x = torch.rand(1,
                           self._dim_in,
                           self._input_size,
                           self._input_size,
                           self._input_size,
                           requires_grad=False)
        else:
            raise NotImplementedError(f'VAEs do not support convolutional '
                                      f'layers of class: {self._conv_layer_class}')

        with torch.no_grad():
            out, _ = self._dcgan_encoder(x)

        return torch.numel(out)

    def _construct(self):
        self._mean_layer = nn.Linear(self._feature_map_size,
                                     self._dim_l)
        self._logvar_layer = nn.Linear(self._feature_map_size,
                                       self._dim_l)

    def forward(self, x):

        # If use_last_layer=False, latent are feature maps
        latent, latents = self._dcgan_encoder(x)
        mean = self._mean_layer(latent)
        logvar = self._logvar_layer(latent)

        return (mean, logvar), latents

    def init_weights(self):
        """
        Weight initialization method
        Returns:
            VAEEncoder with initialized weights

        """
        self._mean_layer.weight.data.fill_(0.01)
        self._logvar_layer.weight.data.fill_(0.01)

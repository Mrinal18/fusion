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
        weights_initialization_type='xavier_uniform'
    ):
        """
        Initialization of base class of convolution layer

        Args:
            :param conv_layer_class: class of convolution layer
            :param conv_layer_args: parameters of convolution layer
            :param norm_layer_class:  class of normalization layer
            :param norm_layer_args: parameters of normalization layer
            :param dp_layer_class: class of droupout layer
            :param dp_layer_args: parameters of droupout layer
            :param activation_class: class of activation function
            :param activation_args: parameters of activation function
            :param weights_initialization_type: type of initialization weights

        :return
            Base class of convolution layer
        """

        super(BaseConvLayer, self).__init__(
            conv_layer_class=conv_layer_class,
            norm_layer_class=norm_layer_class,
            dp_layer_class=dp_layer_class,
            activation_class=activation_class,
            weights_initialization_type=weights_initialization_type
        )
        self._layer = nn.ModuleList()
        print (conv_layer_args)
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
        self.init_weights()

    def forward(self, x):
        """
        Forward method of base class of convolution layer
        Args:
            :param x: tensor
        Returns:
            Args:
                :param x: tensor after forward method
                :param conv_latent: latent presentation of first convolution layer
        """
        x = self._layer[0](x)
        conv_latent = x
        for layer in self._layer[1:]:
            x = layer(x)
        return x, conv_latent

    def init_weights(self):
        """
        Method for initialization weights
        Returns:
            Layer with initialization weights

        """
        if self._weights_initialization_type == 'xavier_uniform':
            nn.init.xavier_uniform_(
                self._layer[0].weight, gain=nn.init.calculate_gain("relu")
            )
            if not isinstance(self._layer[0].bias, type(None)):
                nn.init.constant_(self._layer[0].bias, 0)
        elif self._weights_initialization_type == 'skip':
            pass
        else:
            raise NotImplementedError

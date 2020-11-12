import zope.interface


class IBaseArchitecture(zope.interface.Interface):

    conv_layer_class = zope.interface.Attribute(
        """Class for N-Dim Convolutional layer"""
    )
    norm_layer_class = zope.interface.Attribute(
        """Class of Normalization layer"""
    )
    dp_layer_class = zope.interface.Attribute(
        """Class of Dropout layer"""
    )
    activation_class = zope.interface.Attribute(
        """Class of activation function"""
    )
    weights_initlization_type = zope.interface.Attribute(
        """Name of weights initilization function"""
    )

    def _init_weights():
        """Weight initilization
        """
        pass

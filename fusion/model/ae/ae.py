from fusion.model import AMultiSourceModel
from fusion.model import ModelOutput


class AE(AMultiSourceModel):
    def __init__(self, sources, architecture, architecture_params):
        """
        Initialization class of autoencoder model

        Args:
            :param sources:
            :param architecture: type of architecture
            :param architecture_params: parameters of architecture

        Return:
            Autoencoder model
        """
        super(AE, self).__init__(sources, architecture, architecture_params)

    def _source_forward(self, source_id, x):

        return self._encoder[source_id](x[int(source_id)])

    def forward(self, x):
        """
        Forward method of autoencoder model

        Args:

            :param x: input tensor
        Return:
            Result of forward method of autoencoder model

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs['x'] = {}
        ret.attrs['x_hat'] = {}
        for source_id, _ in self._encoder.items():
            ret.attrs['x'] = x[int(source_id)]
            z, x_hat = self._source_forward(source_id, x)
            ret.z[source_id] = z
            ret.attrs['x_hat'] = x_hat
        return ret

from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class AE(ABaseModel):
    def __init__(self, sources, architecture, architecture_params):
        """

        :param sources:
        :param architecture:
        :param architecture_params:
        """
        super(AE, self).__init__(sources, architecture, architecture_params)

    def _source_forward(self, source_id, x):
        """

        :param source_id:
        :param x:
        :return:
        """
        return self._encoder[source_id](x[int(source_id)])

    def forward(self, x):
        """

        :param x:
        :return:
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

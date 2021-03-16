from fusion.model import AMultiSourceModel
from fusion.model import ModelOutput


class AE(AMultiSourceModel):
    def __init__(self, sources, architecture, architecture_params):
        super(AE, self).__init__(sources, architecture, architecture_params)

    def _source_forward(self, source_id, x):
        return self._encoder[source_id](x[int(source_id)])

    def forward(self, x):
        ret = ModelOutput(z={}, attrs={})
        ret.attrs['x'] = {}
        ret.attrs['x_hat'] = {}
        for source_id, _ in self._encoder.items():
            ret.attrs['x'] = x[int(source_id)]
            z, x_hat = self._source_forward(source_id, x)
            ret.z[source_id] = z
            ret.attrs['x_hat'] = x_hat
        return ret

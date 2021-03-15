import copy
from fusion.architecture.projection_head import ConvHead, LatentHead
from fusion.model import AMultiSourceModel
from fusion.model import ModelOutput
import torch.nn as nn


class Dim(AMultiSourceModel):
    def __init__(
        self,
        sources,
        architecture,
        architecture_params,
        conv_head_params,
        latent_head_params
    ):
        # create encoders for each view
        super(Dim, AMultiSourceModel).__init__(architecture, architecture_params)
        # create convolutional heads
        self._conv_heads = nn.ModuleDict()
        for id_view in self.encoder.keys():
            self._conv_heads[id_view] = nn.ModuleDict()
            for i, dim_conv in architecture_params['dim_cls']:
                conv_head = ConvHead(
                    **conv_head_params
                )
                self._conv_heads[str(source_id)][str(dim_conv)] = conv_head
        # create latent heads
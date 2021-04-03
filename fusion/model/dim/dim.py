import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fusion.architecture.projection_head import ConvHead, LatentHead
from fusion.model import ABaseModel
from fusion.model.misc import  ModelOutput

class Dim(ABaseModel):
    def __init__(
        self,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any],
        conv_head_params: Optional[Dict[str, Any]] = None,
        latent_head_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            sources:
            architecture:
            architecture_params:
            conv_head_params:
            latent_head_params:
        Return:

        """
        # create encoders for each source
        super().__init__(sources, architecture, architecture_params)
        self._input_size = architecture_params['input_size']
        self._conv_layer_class = architecture_params[
            'conv_layer_class'] if 'conv_layer_class' in architecture_params.keys() else nn.Conv2d
        # create convolutional heads
        self._conv_heads = nn.ModuleDict()
        conv_head_params = None
        for source_id in self._encoder.keys():
            self._conv_heads[source_id] = nn.ModuleDict()
            for conv_latent_size in architecture_params['dim_cls']:
                conv_head_params = self._parse_conv_head_params(
                    conv_head_params, architecture_params, conv_latent_size, source_id
                )
                conv_head = ConvHead(**conv_head_params)
                conv_head.init_weights()
                self._conv_heads[str(source_id)][str(conv_latent_size)] = conv_head
        # create latent heads
        self._latent_heads = nn.ModuleDict()
        latent_head_params = None
        for source_id in self._encoder.keys():
            latent_head_params = self._parse_latent_head_params(
                latent_head_params, architecture_params
            )
            latent_head = LatentHead(**latent_head_params)
            latent_head.init_weights()
            self._latent_heads[source_id] = latent_head


    def _source_forward(self, source_id: int, x: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        source_id_s = str(source_id)
        z, latents = self._encoder[source_id_s](x[int(source_id)])
        # pass latents through projection heads
        for conv_latent_size, conv_latent in latents.items():
            if conv_latent_size == 1:
                conv_latent = self._latent_heads[source_id_s](conv_latent)
            elif conv_latent_size > 1:
                conv_latent = self._conv_heads[source_id_s][
                    str(conv_latent_size)](conv_latent)
            else:
                assert False
            latents[conv_latent_size] = conv_latent
        return z, latents

    def forward(self, x: Tensor) -> ModelOutput:
        """

        Args:
            x: input tensor

        Return

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs['latents'] = {}
        for source_id, _ in self._encoder.items():
            source_id = int(source_id)
            z, conv_latents = self._source_forward(source_id, x)
            ret.z[int(source_id)] = z
            ret.attrs['latents'][source_id] = conv_latents
        return ret

    def _parse_conv_head_params(
            self, conv_head_params: Optional[Dict[str, Any]],
            architecture_params: Dict[str, Any],
            conv_latent_size: int, source_id: int
    ) -> Dict[str, Any]:
        if conv_head_params is None:
            # by design choice
            conv_head_params = copy.deepcopy(dict(**architecture_params))
            conv_head_params.pop('dim_cls')
            conv_head_params.pop('input_size')
            dim_in = self._find_dim_in(conv_latent_size, source_id) # find the dim_in for dim_conv
            conv_head_params['dim_in'] = dim_in
            conv_head_params['dim_h'] = conv_head_params['dim_l']
        return conv_head_params

    def _parse_latent_head_params(self, latent_head_params: Optional[Dict[str, Any]],
        architecture_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if latent_head_params is None:
            # by design choice
            latent_head_params = copy.deepcopy(dict(**architecture_params))
            latent_head_params.pop('dim_cls')
            latent_head_params.pop('input_size')
            latent_head_params['dim_in'] = latent_head_params['dim_l']
            latent_head_params['dim_h'] = latent_head_params['dim_l']
        return latent_head_params

    def _find_dim_in(self, conv_latent_size, source_id):
        dim_conv = None
        with torch.no_grad():
            batch_size = 2
            dim_in = 1
            dummy_encoder = self._encoder[source_id].eval()
            if self._conv_layer_class is nn.Conv2d:
                dummy_batch = torch.FloatTensor(
                    batch_size, dim_in, self._input_size, self._input_size)
            elif self._conv_layer_class is nn.Conv3d:
                dummy_batch = torch.FloatTensor(
                    batch_size, dim_in,
                    self._input_size, self._input_size, self._input_size
                )
            else:
                raise NotImplementedError
            x = dummy_batch
            for layer in dummy_encoder.get_layers():
                x, conv_latent = layer(x)
                if conv_latent.size(-1) == conv_latent_size:
                    dim_conv = conv_latent.size(1)
            if dim_conv is None:
                assert False, \
                    f'There is no features with ' \
                    f'convolutional latent size {conv_latent_size} '
        return dim_conv

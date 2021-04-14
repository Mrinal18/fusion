from .vae_encoder import VAEEncoder
from .vae_decoder import VAEDecoder
from fusion.architecture import ABaseArchitecture

from typing import Tuple

import torch


class VAEAutoencoder(ABaseArchitecture):
    def __init__(
        self,
        prior_dist,
        likelihood_dist,
        post_dist,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        ll_scale: torch.tensor,
        pz_params: Tuple[torch.tensor, torch.tensor],
    ):
        super(VAEAutoencoder, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = encoder
        self.dec = decoder
        self._pz_params = pz_params
        self._ll_scale = ll_scale
        self._qz_x_params = None
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    def forward(self, x, K=1):
        self._qz_x_params, enc_latents = self.enc(x)
        z_dist = self.qz_x(*self._qz_x_params)
        #zs = z_dist.rsample(torch.Size([K]))
        zs = z_dist.rsample()
        mean_x, dec_latents = self.dec(zs)
        px_z = self.px_z(mean_x, self._ll_scale)
        return z_dist, px_z, zs, (enc_latents, dec_latents)

    def init_weights(self):
        pass

from .vae_encoder import VAEEncoder
from .dcgan_decoder import DcganDecoder
from fusion.architecture import ABaseArchitecture

import torch


class VAEAutoencoder(ABaseArchitecture):
    def __init__(
        self,
        prior_dist,
        likelihood_dist,
        post_dist,
        enc: VAEEncoder,
        dec: DcganDecoder,
        pz_params,
    ):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = enc
        self.dec = dec
        self._pz_params = pz_params
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
        self._qz_x_params, latents = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu
# Modified work Copyright 2020 Alex Fedorov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from fusion.criterion.loss import ABaseLoss
from fusion.model.misc import ModelOutput

import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional


class CanonicalCorrelation(ABaseLoss):
    def __init__(
        self,
        eps=1e-3,
        r1=1e-7,
        r2=1e-7,
        use_all_singular_values=True,
        num_canonical_components: Optional[int] = None
    ):
        super().__init__()
        self._eps = eps
        self._r1 = r1
        self._r2 = r2
        self._use_all_singular_values = use_all_singular_values
        if not self._use_all_singular_values:
            assert self._num_canonical_components is not None
            assert self._num_canonical_components > 0
        self._num_canonical_components = num_canonical_components

    def forward(self, preds: ModelOutput, target: Optional[Tensor] = None) -> Tensor:
        ret_loss = None
        raw_losses = {}
        for source_id_one, z_one in preds.z.items():
            for source_id_two, z_two in preds.z.items():
                if source_id_one != source_id_two:
                    name = f'CCA_{source_id_one}:{source_id_two}'
                    loss = self._linear_cca(z_one, z_two)
                    raw_losses[name] = loss.item()
                    ret_loss = ret_loss + loss if ret_loss is not None else loss
        return ret_loss, raw_losses

    def _linear_cca(self, h1, h2):
        # Transpose matrices so each column is a sample
        h1, h2 = h1.t(), h2.t()

        o1 = o2 = h1.size(0)
        m = h1.size(1)

        h1_bar = h1 - h1.mean(dim=1).unsqueeze(dim=1)
        h2_bar = h2 - h2.mean(dim=1).unsqueeze(dim=1)
        # Compute covariance matrices and add diagonal so they are
        # positive definite
        sigma_hat12 = (1.0 / (m - 1)) * torch.matmul(h1_bar, h2_bar.t())
        sigma_hat11 = (1.0 / (m - 1)) * torch.matmul(h1_bar, h1_bar.t()) + \
            self._r1 * torch.eye(o1, device=h1_bar.device)
        sigma_hat22 = (1.0 / (m - 1)) * torch.matmul(h2_bar, h2_bar.t()) + \
            self._r2 * torch.eye(o2, device=h2_bar.device)

        # Calculate the root inverse of covariance matrices by using
        # eigen decomposition
        [d1, v1] = torch.symeig(
            sigma_hat11, eigenvectors=True)
        [d2, v2] = torch.symeig(
            sigma_hat22, eigenvectors=True)

        # Additional code to increase numerical stability
        pos_ind1 = torch.gt(d1, self._eps).nonzero()[:, 0]
        d1 = d1[pos_ind1]
        v1 = v1[:, pos_ind1]
        pos_ind2 = torch.gt(d2, self._eps).nonzero()[:, 0]
        d2 = d2[pos_ind2]
        v2 = v2[:, pos_ind2]

         # Compute sigma hat matrices using the edited covariance matrices
        sigma_hat11_root_inv = torch.matmul(
            torch.matmul(v1, torch.diag(d1 ** -0.5)), v1.t())
        sigma_hat22_root_inv = torch.matmul(
            torch.matmul(v2, torch.diag(d2 ** -0.5)), v2.t())

        # Compute the T matrix, whose matrix trace norm is the loss
        tval = torch.matmul(
            torch.matmul(sigma_hat11_root_inv, sigma_hat12),
            sigma_hat22_root_inv
        )

        if self._use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(tval.t(), tval))
            corr = torch.sqrt(tmp)
        else:
            # just the top self._num_canonical_components singular values are used
            u, v = torch.symeig(
                torch.matmul(
                    tval.t(), tval), eigenvectors=True)
            u = u.topk(self._num_canonical_components)[0]
            corr = torch.sum(torch.sqrt(u))
        return -corr

import abc
from collections import namedtuple
from fusion.criterion.loss import ABaseLoss
from fusion.criterion.loss.dim import dim_mode_provider
import torch
import torch.nn as nn


RR_MODE = 'RR'
CR_MODE = 'CR'
XX_MODE = 'XX'
CC_MODE = 'CC'


class MultiDim(ABaseLoss):
    def __init__(
        self,
        dim_cls,
        estimator,
        modes=[CR_MODE, XX_MODE, CC_MODE, RR_MODE],
        trade_offs=[1., 1., 1., 1.],
    ):
        super(MultiDim, self).__init__()
        assert len(modes) == len(trade_offs)
        self._dim_cls = dim_cls
        self._modes = modes
        self._masks = self._create_masks()
        self._objectives = {}
        for i, mode in enumerate(modes):
            dim_mode_args = {
                'estimator': estimator,
                'trade_off': trade_offs[i],
            }
            self._objectives[mode] = dim_mode_provider(
                mode, **dim_mode_args
            )

    @abc.abstractmethod
    def _create_masks(self):
        pass

    @staticmethod
    def _reshape_target(self, target):
        return target.reshape(target.size(0), target.size(1), -1)

    @staticmethod
    def _sample_location(conv_latents, masks):
        n_batch = conv_latents.size(0)
        n_channels = conv_latents.size(1)
        if masks is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, masks.size(0), (n_batch,))
            if torch.cuda.is_available():
                mask_idx = mask_idx.cuda(torch.device("cuda:{}".format(0)))
                masks = masks.cuda()
            # print (r_cnv.size(), mask_idx.size(), masks[mask_idx].size())
            conv_latents = torch.masked_select(conv_latents, masks[mask_idx])
        # flatten features for use as globals in glb->lcl nce cost
        locations = conv_latents.reshape(n_batch, n_channels, 1)
        return locations

    def _prepare_sources_targets(self, latents):
        sources, targets = {}, {}
        for source_id in latents.keys():
            sources[source_id] = {}
            targets[source_id] = {}
            for conv_latent_size in latents[source_id].keys():
                if conv_latent_size == 1:
                    source = self._sample_location(
                        latents[source_id][conv_latent_size],
                        masks=None
                    )
                    sources[source_id][conv_latent_size] = source
                elif conv_latent_size > 1:
                    source = self._sample_location(
                        latents[source_id][conv_latent_size],
                        masks=self._masks[conv_latent_size]
                    )
                    sources[source_id][conv_latent_size] = source
                    target = self._reshape_target(
                        latents[source_id][conv_latent_size]
                    )
                    targets[source_id][conv_latent_size] = target
                else:
                    assert conv_latent_size < 0
        return sources, targets

    def forward(self, preds, target=None):
        del target
        # prepare sources and targets
        latents = preds.attrs['latents']
        sources, targets = self._prepare_sources_targets(latents)
        # compute losses
        ret_loss = 0
        raw_losses = {}
        for _, objective in self._objectives.items():
            loss, raw = objective(sources, targets)
            ret_loss = ret_loss + loss
            raw_losses.update(raw)
        return ret_loss, raw_losses


class SpatialMultiDim(MultiDim):
    def __init__(self):
        super(SpatialMultiDim, self).__init__()

    def _create_masks(self):
        masks = {}
        for dim_cl in self._dim_cls:
            mask = torch.zeros((dim_cl, dim_cl, 1, dim_cl, dim_cl))
            for i in range(dim_cl):
                for j in range(dim_cl):
                    mask[i, j, 0, i, j] = 1
            mask = torch.BoolTensor(mask)
            mask = mask.reshape(-1, 1, dim_cl, dim_cl)
            masks[dim_cl] = nn.Parameter(mask, requires_grad=False)
            if torch.cuda.is_available():
                masks[dim_cl].cuda()
        return masks


class VolumetricMultiDim(MultiDim):
    def __init__(self):
        super(VolumetricMultiDim, self).__init__()

    def _create_masks(self):
        masks = {}
        for dim_cl in self._dim_cls:
            mask = torch.zeros((dim_cl, dim_cl, dim_cl, 1, dim_cl, dim_cl, dim_cl))
            for i in range(dim_cl):
                for j in range(dim_cl):
                    for k in range(dim_cl):
                        mask[i, j, k, 0, i, j, k] = 1
            mask = torch.BoolTensor(mask)
            mask = mask.reshape(-1, 1, dim_cl, dim_cl, dim_cl)
            masks[dim_cl] = nn.Parameter(mask, requires_grad=False)
        return masks


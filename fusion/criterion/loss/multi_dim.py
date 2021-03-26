import abc
from fusion.criterion.loss import ABaseLoss, MAXIMIZE
from fusion.criterion.loss.dim import dim_mode_provider
import torch
import torch.nn as nn


RR_MODE = 'RR'
CR_MODE = 'CR'
XX_MODE = 'XX'
CC_MODE = 'CC'

class MultiDim(ABaseLoss):

    @abc.abstractmethod
    def __init__(
        self,
        dim_cls,
        estimator,
        estimator_args,
        critic,
        critic_args,
        clip,
        clip_args,
        modes=[CR_MODE, XX_MODE, CC_MODE, RR_MODE],
        direction=MAXIMIZE
    ):
        super(MultiDim, self).__init__()
        self._dim_cls = dim_cls
        self._modes = modes
        self._masks = self._create_masks()
        self.

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

    def forward(self, input, target):
        del target



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


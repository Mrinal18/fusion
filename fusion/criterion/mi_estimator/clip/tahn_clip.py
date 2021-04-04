import torch
from torch import Tensor

from fusion.criterion.mi_estimator.clip import ABaseClip


class TahnClip(ABaseClip):
    def __call__(self, scores: Tensor) -> Tensor:
        if self._clip_value is not None:
            clipped = torch.tanh((1.0 / self._clip_value) * scores)
            clipped = self._clip_value * clipped
        else:
            clipped = scores
        return clipped



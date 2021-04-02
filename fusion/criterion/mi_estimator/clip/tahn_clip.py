from fusion.criterion.mi_estimator.clip import ABaseClip
import torch


class TahnClip(ABaseClip):
    def __call__(self, scores):
        if self._clip_value is not None:
            clipped = torch.tanh((1.0 / self._clip_value) * scores)
            clipped = self._clip_value * clipped
        else:
            clipped = scores
        return clipped



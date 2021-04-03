import abc
from typing import Optional


class BaseDim(abc.ABC):
    _name: Optional[str] = None

    def __init__(
        self,
        estimator,
        weight: float = 1.0,
    ):
        self._estimator = estimator
        self._weight = weight

    @abc.abstractmethod
    def __call__(self, reps, convs):
        pass

    def _update_loss(self, name, ret_loss, raw_losses, loss, penalty):
        loss = self._weight * loss
        raw_losses[f'{name}_loss'] = loss.item()
        ret_loss = ret_loss + loss if ret_loss is not None else loss
        if penalty is not None:
            raw_losses[f'{name}_penalty'] = penalty.item()
            ret_loss = ret_loss + penalty if ret_loss is not None else penalty
        return ret_loss, raw_losses
from torch import Tensor

from fusion.criterion.mi_estimator.penalty import ABasePenalty


class L2Penalty(ABasePenalty):
    def __init__(self, weight: float = 4e-2):
        self._weight = weight

    def __call__(self, scores: Tensor) -> Tensor:
        penalty = scores ** 2.0
        penalty = penalty.mean()
        penalty *= self._weight
        return penalty

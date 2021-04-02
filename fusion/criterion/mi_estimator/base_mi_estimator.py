import abc
from typing import Optional, Tuple

from torch import Tensor

from fusion.criterion.mi_estimator.critic import critic_provider
from fusion.criterion.mi_estimator.clip import clip_provider
from fusion.criterion.mi_estimator.penalty import penalty_provider


class ABaseMIEstimator(abc.ABC):
    def __init__(self, critic_setting, clip_setting=None, penalty_setting=None):
        args = {} if critic_setting.args is None else critic_setting.args
        self._critic = critic_provider.get(
            critic_setting.class_type, **args
        )
        self._clip = None
        self._penalty = None
        if clip_setting.class_type is not None:
            args = {} if clip_setting.args is None else clip_setting.args
            self._clip = clip_provider.get(
                clip_setting.class_type, **args
            )
        if penalty_setting.class_type is not None:
            args = {} if penalty_setting.args is None else penalty_setting.args
            self._penalty = penalty_provider.get(
                penalty_setting.class_type, **args
            )

    @abc.abstractmethod
    def __call__(self, x: Tensor, y: Tensor):
        pass

    def _compute_scores(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        # bs x dim_l x locations -> bs x locations x dim_l
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        # bs*locations x dim_l
        x = x.reshape(-1, dim_l)
        y = y.reshape(-1, dim_l)

        # Outer product bs*xlocs*ylocs x bs*xlocs*ylocs
        scores = self._critic(y, x)
        penalty = None
        if self._penalty is not None:
            penalty = self._penalty(scores)
        if self._clip is not None:
            scores = self._clip(scores)

        # bs*bs*xlocs*ylocs -> bs x y_locs x bs x x_locs
        scores = scores.reshape(bs, y_locs, bs, x_locs)
        # bs x bs x x_locs x y_locs tensor.
        scores = scores.permute(0, 2, 3, 1)
        return scores, penalty

    @staticmethod
    def _check_input(x: Tensor, y: Tensor):
        assert len(x.size()) == 3
        assert len(y.size()) == 3
        assert x.size(0) == y.size(0)
        assert x.size(1) == y.size(1)

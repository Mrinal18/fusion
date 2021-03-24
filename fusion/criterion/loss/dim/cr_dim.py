from fusion.criterion.loss import MAXIMIZE
from fusion.criterion.loss.dim import BaseDim


class CrDim(BaseDim):
    def __init__(
        self,
        estimator,
        estimator_args,
        critic,
        critic_args,
        clip,
        clip_args,
        direction=MAXIMIZE
    ):
        super(CrDim, self).__init__(
            estimator, estimator_args, critic, critic_args,
            clip, clip_args, direction=MAXIMIZE,
        )

    def compute_scores(self, x, y, mask_mat):
        pass

import abc
from fusion.criterion.loss import MAXIMIZE
import torch.nn as nn


class BaseDim(abc.ABC):
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
        self._estimator =

    @abc.abstractmethod
    def compute_scores(self, x, y, mask_mat):
        pass


import abc
from fusion.criterion.loss import MAXIMIZE
import torch.nn as nn


class BaseDim(abc.ABC):
    def __init__(
        self,
        estimator,
        trade_off=1,
    ):
        self._estimator = estimator
        self._trade_off = trade_off

    @abc.abstractmethod
    def compute_scores(self, x, y, mask_mat):
        pass


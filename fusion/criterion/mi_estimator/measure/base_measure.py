import abc
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def log_sum_exp(x: Tensor, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


class ABaseMeasure(abc.ABC):
    def __init__(self, average=False):
        self._average = average

    @abc.abstractmethod
    def get_positive_expectation(self, p):
        pass

    @abc.abstractmethod
    def get_negative_expectation(self, q):
        pass

    def _if_average(self, e):
        return e.mean() if self._average else e


class GanMeasure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = -F.softplus(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = F.softplus(-q) + q
        return self._if_average(Eq)


class JsdMeasure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = math.log(2.) - F.softplus(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = F.softplus(-q) + q - math.log(2.)
        return self._if_average(Eq)


class X2Measure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = p ** 2
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = -0.5 * ((torch.sqrt(q ** 2) + 1.) ** 2)
        return self._if_average(Eq)


class KLMeasure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = torch.exp(q - 1.)
        return self._if_average(Eq)


class RKLMeasure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = -torch.exp(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = q - 1.
        return self._if_average(Eq)


class DVMeasure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = log_sum_exp(q, 0) - math.log(q.size(0))
        return self._if_average(Eq)


class H2Measure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = 1. - torch.exp(-p)
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = torch.exp(q) - 1.
        return self._if_average(Eq)


class W1Measure(ABaseMeasure):
    def get_positive_expectation(self, p):
        Ep = p
        return self._if_average(Ep)

    def get_negative_expectation(self, q):
        Eq = q
        return self._if_average(Eq)

from fusion.criterion.mi_estimator.critic import ABaseCritic
import torch


class SeparableCritic(ABaseCritic):
    def __call__(self, x, y):
        self._check(x, y)
        s = torch.mm(x, y.t())
        return s

    @staticmethod
    def _check(x, y):
        assert len(x.size()) == 2
        assert len(y.size()) == 2
        assert x.size(1) == y.size(1)


class ScaledDotProduct(SeparableCritic):
    def __call__(self, x, y):
        print (x.max())
        print (y.max())
        s = super().__call__(x, y)
        print (s.max())
        dim_l = x.size(1)
        s = s / dim_l ** 0.5
        return s


class CosineSimilarity(SeparableCritic):
    def __init__(self, temperature=1.):
        self._temperature = temperature

    def __call__(self, x, y):
        s = super().__call__(x, y)
        dim_l = x.size(1)
        x_norm = torch.norm(x, dim=1)
        y_norm = torch.norm(y, dim=1)
        s = s / x_norm
        s = s / y_norm
        s = s / self._temperature
        return s

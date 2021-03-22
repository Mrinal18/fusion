from fusion.criterion.mi_estimator.critic import ABaseCritic
import torch


class SeparableCritic(ABaseCritic):
    def __init__(self, **critic_attrs):
        super(SeparableCritic, self).__init__(**critic_attrs)

    def score(self, x, y):
        pass

    @staticmethod
    def _check(x, y):
        assert len(x.size()) == 2
        assert len(y.size()) == 2
        assert x.size(1) == y.size(1)


class ScaledDotProduct(SeparableCritic):
    def score(self, x, y):
        self._check(x, y)
        s = torch.mm(x, y.t())
        dim_l = x.size(1)
        s = s / dim_l ** 0.5
        return s


class CosineSimilarity(SeparableCritic):
    def __init__(self, temperature=1., **critic_attrs):
        super(SeparableCritic, self).__init__(**critic_attrs)
        self._temperature = temperature

    def score(self, x, y):
        self._check(x, y)
        s = torch.mm(x, y.t())
        x_norm = torch.norm(x, dim=1)
        y_norm = torch.norm(y, dim=1)
        s = s / x_norm
        s = s / y_norm
        s = s / self._temperature
        return s

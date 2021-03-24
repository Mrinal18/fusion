from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.mi_estimator.measure import measure_provider
import torch


class FenchelDualEstimator(ABaseMIEstimator):
    def __init__(self, critic, penalty=None, measure='JSD'):
        super(FenchelDualEstimator, self).__init__(critic, penalty=penalty)
        self._measure = measure_provider.get(measure, **{})

    def __call__(self, x, y):
        assert len(x.size()) == 3
        assert len(y.size()) == 3
        assert x.size(0) == y.size(0), ''
        assert x.size(1) == y.size(1)

        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        # BS x Dim L x locations -> BS x locations x Dim L
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        # BS \times locations x Dim L
        x = x.reshape(-1, dim_l)
        y = y.reshape(-1, dim_l)

        # Outer product
        scores = self._critic(y, x)
        penalty = None
        if self._penalty is not None:
            penalty = self._penalty(scores)
        # we want a N x N x n_local x n_multi tensor.
        scores = scores.reshape(bs, y_locs, bs, x_locs)
        scores = scores.permute(0, 2, 3, 1)

        mask = torch.eye(bs).to(x.device)
        n_mask = 1 - mask

        e_pos = self._measure.get_positive_expectation(scores)
        e_pos = e_pos.mean(2).mean(2)
        e_pos = (e_pos * mask).sum() / mask.sum()

        e_neg = self._measure.get_negative_expectation(scores)
        e_neg = e_neg.mean(2).mean(2)
        e_neg = (e_pos * mask).sum() / mask.sum()

        loss = e_neg - e_pos
        return loss, penalty

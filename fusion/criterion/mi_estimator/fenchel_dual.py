from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.mi_estimator.measure import measure_provider
import torch


class FenchelDualEstimator(ABaseMIEstimator):
    def __init__(self, critic, clip=None, penalty=None, measure='JSD'):
        super(FenchelDualEstimator, self).__init__(critic, clip, penalty=penalty)
        self._measure = measure_provider.get(measure, **{})

    def __call__(self, x, y):
        self._check_input(x, y)

        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        scores, penalty = self._compute_scores(x, y)

        pos_mask = torch.eye(bs)
        pos_mask = pos_mask.to(x.device)
        neg_mask = 1 - pos_mask

        e_pos = self._measure.get_positive_expectation(scores)
        e_pos = e_pos.mean(2).mean(2)
        e_pos = (e_pos * pos_mask).sum() / pos_mask.sum()

        e_neg = self._measure.get_negative_expectation(scores)
        e_neg = e_neg.mean(2).mean(2)
        e_neg = (e_neg * neg_mask).sum() / neg_mask.sum()

        loss = e_neg - e_pos
        return loss, penalty

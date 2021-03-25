from fusion.criterion.mi_estimator import ABaseMIEstimator
import torch


class InfoNceEstimator(ABaseMIEstimator):
    def __call__(self, x, y):
        assert self._clip is not None
        self._check_input(x, y)

        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        scores, penalty = self._compute_scores(x, y)

        # bs x bs
        pos_mask = torch.eye(bs)
        # bs x bs x 1
        pos_mask = pos_mask.unsqueeze(2)
        # bs x bs x x_locs
        pos_mask = pos_mask.expand(-1, -1, x_locs)
        # bs x bs x x_locs x 1
        pos_mask = pos_mask.unsqueeze(3)
        # bs x bs x x_locs x y_locs
        pos_mask = pos_mask.expand(-1, -1, -1, y_locs).float()
        pos_mask = pos_mask.to(x.device)

        # bs x bs x x_locs x y_locs
        pos_scores = (pos_mask * scores)
        pos_scores = pos_scores.reshape(bs, bs, -1)
        pos_scores = pos_scores.sum(1)

        neg_mask = 1 - pos_mask
        # bs x bs x x_locs x y_locs
        neg_scores = (neg_mask * scores)
        # mask self-examples
        #neg_scores -= 10 * pos_mask
        neg_scores -= self._clip.clip_value * pos_mask
        neg_scores = neg_scores.reshape(bs, -1)
        neg_mask = neg_mask.reshape(bs, -1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]
        neg_sumexp = (
                neg_mask * torch.exp(neg_scores - neg_maxes)
        ).sum(dim=1, keepdim=True)
        all_logsumexp = torch.log(
            torch.exp(pos_scores - neg_maxes) + neg_sumexp)
        pos_shiftexp = pos_scores - neg_maxes
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()
        return nce_scores, penalty

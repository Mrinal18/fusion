from fusion.criterion.mi_estimator import ABaseMIEstimator
import torch


class InfoNceEstimator(ABaseMIEstimator):
    def __call__(self, x, y):
        assert len(x.size()) == 3
        assert len(y.size()) == 3
        assert x.size(0) == y.size(0)
        assert x.size(1) == y.size(1)

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
        assert self._clip is not None
        scores = self._clip(scores)

        # bs*bs*xlocs*ylocs -> bs x y_locs x bs x x_locs
        scores = scores.reshape(bs, y_locs, bs, x_locs)
        # bs x bs x x_locs x y_locs tensor.
        scores = scores.permute(0, 2, 3, 1)

        # bs x bs
        mask = torch.eye(bs)
        # bs x bs x 1
        mask = mask.unsqueeze(2)
        # bs x bs x x_locs
        mask = mask.expand(-1, -1, x_locs)
        # bs x bs x x_locs x 1
        mask = mask.unsqueeze(3)
        # bs x bs x x_locs x y_locs
        mask = mask.expand(-1, -1, -1, y_locs).float()
        mask = mask.to(x.device)

        pos_mask = mask
        # bs x bs x x_locs x y_locs
        pos_scores = (pos_mask * scores)
        pos_scores = pos_scores.reshape(bs, bs, -1)
        pos_scores = pos_scores.sum(1)

        neg_mask = 1 - mask
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

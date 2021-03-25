import abc


class ABaseMIEstimator(abc.ABC):
    def __init__(self, critic, clip=None, penalty=None):
        self._critic = critic
        self._penalty = penalty
        self._clip = clip

    @abc.abstractmethod
    def __call__(self, x, y):
        pass

    def _compute_scores(self, x, y):
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
        if self._clip is not None:
            scores = self._clip(scores)

        # bs*bs*xlocs*ylocs -> bs x y_locs x bs x x_locs
        scores = scores.reshape(bs, y_locs, bs, x_locs)
        # bs x bs x x_locs x y_locs tensor.
        scores = scores.permute(0, 2, 3, 1)
        return scores, penalty

    def _check_input(self, x, y):
        assert len(x.size()) == 3
        assert len(y.size()) == 3
        assert x.size(0) == y.size(0)
        assert x.size(1) == y.size(1)
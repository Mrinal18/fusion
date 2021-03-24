from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.mi_estimator.measure import measure_provider


class FenchelDualEstimator(ABaseMIEstimator):
    def __init__(self, critic, penalty=None, measure='JSD'):
        super(FenchelDualEstimator, self).__init__(critic, penalty=penalty)
        self._measure = measure_provider.get(measure, **{})

    def __call__(self, x, y):
        assert len(x.size()) == 2
        assert len(y.size()) == 2
        assert x.size(1) == y.size(1)

        scores = self._critic(x, y)
        penalty = None
        if self._penalty is not None:
            penalty = self._penalty(scores)










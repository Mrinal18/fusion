from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.mi_estimator.measure import measure_provider


class FenchelDualEstimator(ABaseMIEstimator):
    def __init__(self, critic, penalty=None, measure='JSD'):
        super(FenchelDualEstimator, self).__init__(critic, penalty=penalty)
        self._measure = measure_provider.get(measure, **{})

    def __call__(self, x, y):
        # TODO: implement this
        pass


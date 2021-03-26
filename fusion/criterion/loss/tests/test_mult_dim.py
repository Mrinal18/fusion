from fusion.model import Dim
from fusion.criterion.mi_estimator import InfoNceEstimator
from fusion.criterion.mi_estimator.critic import SeparableCritic
from fusion.criterion.mi_estimator.clip import TahnClip
from fusion.criterion.mi_estimator.penalty import L2Penalty
from fusion.criterion.loss import SpatialMultiDim
import torch
import unittest


class TestSpatialMultiDim(unittest.TestCase):
    @staticmethod
    def _generate_output():
        dim_in = 1
        dim_l = 4
        dim_cls = [8]
        input_size = 32
        architecture = 'DcganEncoder'
        architecture_params = dict(
            input_size=input_size,
            dim_in=[dim_in, dim_in],
            dim_h=2,
            dim_l=dim_l,
            dim_cls=dim_cls
        )
        sources = [0, 1]
        batch_size = 2
        # create model
        model = Dim(sources, architecture, architecture_params)
        # create input
        x = []
        for _ in sources:
            x.append(torch.rand(batch_size, dim_in, input_size, input_size))
        # forward pass
        output = model(x)
        return output

    def test_spatial_multi_dim(self):
        output = self._generate_output()
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator_type = InfoNceEstimator(critic, clip, penalty=penalty)



if __name__ == '__main__':
    unittest.main()

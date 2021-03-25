from fusion.criterion.mi_estimator import InfoNceEstimator
from fusion.criterion.mi_estimator import FenchelDualEstimator
from fusion.criterion.mi_estimator.critic import SeparableCritic
from fusion.criterion.mi_estimator.clip import TahnClip
from fusion.criterion.mi_estimator.penalty import L2Penalty
import torch
import unittest


class TestMIEstimators(unittest.TestCase):
    @staticmethod
    def _generate_data():
        batch_size = 8
        dim_l = 64
        locs = 32
        zs = 1
        torch.manual_seed(42)
        x = torch.rand(batch_size, dim_l, locs)
        y = torch.rand(batch_size, dim_l, zs)
        return x, y

    def test_infonce_estimator(self):
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator = InfoNceEstimator(critic, clip, penalty=penalty)
        x, y = self._generate_data()
        score, penalty = estimator(x, y)
        score = score.item()
        penalty = penalty.item()
        self.assertAlmostEqual(score, 5.4309, places=4)
        self.assertAlmostEqual(penalty, 10.2132, places=3)

    def test_fenchel_dual(self):
        critic = SeparableCritic()
        penalty = L2Penalty()
        estimator = FenchelDualEstimator(critic, penalty=penalty)
        x, y = self._generate_data()
        score, penalty = estimator(x, y)
        score = score.item()
        penalty = penalty.item()
        self.assertAlmostEqual(score, 14.4711, places=3)
        self.assertAlmostEqual(penalty, 10.2132, places=3)


if __name__ == '__main__':
    unittest.main()

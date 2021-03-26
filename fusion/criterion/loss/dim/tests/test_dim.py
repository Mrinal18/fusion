from fusion.criterion.loss.dim import CrDim, CcDim, RrDim, XxDim
from fusion.criterion.mi_estimator import InfoNceEstimator
from fusion.criterion.mi_estimator.critic import SeparableCritic
from fusion.criterion.mi_estimator.clip import TahnClip
from fusion.criterion.mi_estimator.penalty import L2Penalty
import torch
import unittest


class TestDim(unittest.TestCase):
    @staticmethod
    def _generate_reps_convs():
        torch.manual_seed(42)
        sources = [0, 1]
        conv_latent_size = [32, 1]
        batch_size = 8
        dim_l = 64
        convs = {}
        reps = {}
        for source_id in sources:
            reps[source_id] = {}
            convs[source_id] = {}
            for dim_conv in conv_latent_size:
                locations = dim_conv
                data = torch.rand(batch_size, dim_l, locations)
                if dim_conv == 1:
                    reps[source_id][dim_conv] = data
                else:
                    reps[source_id][dim_conv] = torch.rand(batch_size, dim_l, 1)
                    convs[source_id][dim_conv] = data
        return convs, reps

    def test_cr_dim(self):
        convs, reps = self._generate_reps_convs()
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator = InfoNceEstimator(critic, clip, penalty=penalty)
        objective = CrDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]].item(), 5.4389, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]].item(), 10.7434, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]].item(), 5.4361, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]].item(), 10.4492, places=3)
        self.assertAlmostEqual(loss.item(), 32.0676, places=3)

    def test_cc_dim(self):
        convs, reps = self._generate_reps_convs()
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator = InfoNceEstimator(critic, clip, penalty=penalty)
        objective = CcDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]].item(), 5.4396, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]].item(), 10.2893, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]].item(), 5.4479, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]].item(), 10.3638, places=3)
        self.assertAlmostEqual(loss.item(), 31.5406, places=3)

    def test_xx_dim(self):
        convs, reps = self._generate_reps_convs()
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator = InfoNceEstimator(critic, clip, penalty=penalty)
        objective = XxDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]].item(), 5.4481, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]].item(), 10.7993, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]].item(), 5.4465, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]].item(), 10.3911, places=3)
        self.assertAlmostEqual(loss.item(), 32.0850, places=3)

    def test_rr_dim(self):
        convs, reps = self._generate_reps_convs()
        critic = SeparableCritic()
        clip = TahnClip()
        penalty = L2Penalty()
        estimator = InfoNceEstimator(critic, clip, penalty=penalty)
        objective = RrDim(estimator=estimator, weight=1)
        loss, raw_losses = objective(reps, convs)
        raw_keys = list(raw_losses.keys())
        self.assertAlmostEqual(raw_losses[raw_keys[0]].item(), 1.7178, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[1]].item(), 12.3758, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[2]].item(), 1.6673, places=3)
        self.assertAlmostEqual(raw_losses[raw_keys[3]].item(), 11.3357, places=3)
        self.assertAlmostEqual(loss.item(), 27.0966, places=3)


if __name__ == '__main__':
    unittest.main()

import torch
import unittest

from fusion.model import Dim
from fusion.utils import Setting
from fusion.criterion.loss import SpatialMultiDim
from fusion.criterion.loss.dim import CR_MODE, XX_MODE, RR_MODE, CC_MODE


class TestSpatialMultiDim(unittest.TestCase):
    @staticmethod
    def _generate_output():
        torch.manual_seed(42)
        dim_in = 1
        dim_l = 64
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
        batch_size = 8
        # create model
        model = Dim(sources, architecture, architecture_params)
        # create input
        x = []
        for _ in sources:
            x.append(torch.rand(batch_size, dim_in, input_size, input_size))
        # forward pass
        output = model(x)
        return output, dim_cls

    def test_spatial_multi_dim(self):
        output, dim_cls = self._generate_output()
        critic_setting = Setting(class_type='SeparableCritic', args={})
        clip_setting = Setting(class_type='TahnClip', args={})
        penalty_setting = Setting(class_type='L2Penalty', args={})
        estimator_setting = Setting(
            class_type='InfoNceEstimator',
            args={
                'critic_setting': critic_setting,
                'clip_setting': clip_setting,
                'penalty_setting': penalty_setting
            }
        )
        criterion = SpatialMultiDim(
            dim_cls=dim_cls,
            estimator_setting=estimator_setting,
            modes=[CR_MODE, XX_MODE, CC_MODE, RR_MODE],
            weights=[1., 1., 1., 1.]
        )
        ret_loss, raw_losses = criterion(output)
        losses = [
            9.4785, 0.3428, 9.5307, 0.5374, # CR
            9.9195, 0.5636, 11.8387, 1.0671, # XX
            11.5989, 1.8206, 12.1579, 2.2010, # CC
            0.1648, 1.6105, 0.2620, 2.9762 # RR
        ]
        for i, (_, loss) in enumerate(raw_losses.items()):
            self.assertAlmostEqual(loss.item(), losses[i], places=3)


if __name__ == '__main__':
    unittest.main()
from torch.utils import data
from fusion.dataset.two_view_mnist.two_view_mnist import TwoViewMnist
import unittest

class TestBaseConvLayer(unittest.TestCase):
    def test_forward(self):
        dataset = TwoViewMnist(
            # TODO: Here hard coded path for the dataset
            dataset_dir='/Users/afedorov/Research/data/MNIST',
            batch_size=1
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader('train')), 48000)
        self.assertEqual(len(dataset.get_loader('val')), 12000)
        self.assertEqual(len(dataset.get_loader('test')), 10000)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)

if __name__ == '__main__':
    unittest.main()

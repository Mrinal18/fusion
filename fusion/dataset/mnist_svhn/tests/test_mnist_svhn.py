from fusion.dataset.mnist_svhn.mnist_svhn import MnistSvhn
import unittest

class TestBaseConvLayer(unittest.TestCase):
    def test_forward(self):
        dataset = MnistSvhn(
            # TODO: Here hard coded path for the dataset
            dataset_dir='../../../../data',
            batch_size=1
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader('train')), 1345632)
        self.assertEqual(len(dataset.get_loader('val')), 336408)
        self.assertEqual(len(dataset.get_loader('test')), 300000)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)


if __name__ == '__main__':
    unittest.main()

from fusion.dataset.two_view_mnist.two_view_mnist import TwoViewMnist
import unittest


class TestTwoViewMnist(unittest.TestCase):
    @unittest.skip("Skipping TwoViewMnist, as it requires data loading")
    def test_two_view_mnist(self):
        dataset = TwoViewMnist(
            # TODO: Here hard coded path for the dataset
            dataset_dir='./data/',
            batch_size=1,
            num_workers=1
        )
        dataset.load()
        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader('train')), 48000)
        self.assertEqual(len(dataset.get_loader('valid')), 12000)
        self.assertEqual(len(dataset.get_loader('infer')), 10000)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)


if __name__ == '__main__':
    unittest.main()

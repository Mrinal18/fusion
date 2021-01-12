from fusion.dataset.mnist_svhn.mnist_svhn import MnistSvhn
import unittest

class TestMnistSvhn(unittest.TestCase):
    def test_forward(self):
        dataset = MnistSvhn(
            # TODO: Here hard coded path for the dataset
            dataset_dir='/Users/afedorov/Research/code/fusion/data/',
            batch_size=8,
            views = [0, 1],
            shuffle=False,
            num_workers=2
        )
        dataset.load()
        for set_id in ['train']:
            d = dataset.get_loader(set_id)
            for i, sample in enumerate(d):
                break
            print(sample[0][1])
            print(sample[1][1])
            self.assertEqual((sample[0][1] == sample[1][1]).all(), True)

        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader('train')), 1345620)
        self.assertEqual(len(dataset.get_loader('valid')), 336420)
        self.assertEqual(len(dataset.get_loader('infer')), 300000)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)


if __name__ == '__main__':
    unittest.main()

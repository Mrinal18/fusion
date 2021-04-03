from fusion.dataset.mnist_svhn.mnist_svhn import MnistSvhn
import unittest

class TestMnistSvhn(unittest.TestCase):
    def test_forward(self):
        BATCH_SIZE = 8
        dataset = MnistSvhn(
            # TODO: Here hard coded path for the dataset
            dataset_dir='../../../../data/',
            batch_size=BATCH_SIZE,
            sources = [0, 1],
            shuffle=True,
            num_workers=1,
            drop_last=False
        )
        dataset.load()
        for set_id in ['infer', 'train', 'valid']:
            d = dataset.get_loader(set_id)
            for i, sample in enumerate(d):
                break
            self.assertEqual((sample[0][1] == sample[1][1]).all(), True)

        self.assertEqual(dataset.num_classes, 10)
        self.assertEqual(len(dataset.get_loader('train')), 1345620 // BATCH_SIZE + 1)
        self.assertEqual(len(dataset.get_loader('valid')), 336420 // BATCH_SIZE + 1)
        self.assertEqual(len(dataset.get_loader('infer')), 300000 // BATCH_SIZE)
        self.assertEqual(len(dataset.get_cv_loaders()), 2)
        self.assertEqual(len(dataset.get_all_loaders()), 3)


if __name__ == '__main__':
    unittest.main()

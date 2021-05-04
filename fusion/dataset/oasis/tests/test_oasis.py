from fusion.dataset.oasis.oasis import Oasis
import unittest


class TestOasis(unittest.TestCase):
    @unittest.skip("Skipping MnistSvhn, as it requires data loading")
    def test_oasis(self):
        pass


if __name__ == '__main__':
    unittest.main()

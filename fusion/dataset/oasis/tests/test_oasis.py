from fusion.dataset.oasis.oasis import Oasis
import unittest


class TestOasis(unittest.TestCase):
    @unittest.skip("Skipping Oasis, as it requires OASIS which is not open_sourced")
    def test_oasis(self):
        dataset_dir = "/data/users2/afedorov/trends/oasis/DSLW2/"
        dataset = Oasis(dataset_dir=dataset_dir)


if __name__ == "__main__":
    unittest.main()

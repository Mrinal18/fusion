from fusion.criterion.mi_estimator.critic import CosineSimilarity
from fusion.criterion.mi_estimator.critic import ScaledDotProduct
import torch
import unittest


class TestSeparableCritics(unittest.TestCase):

    def test_scaled_dot_product(self):
        batch_size = 2
        dim_l = 32
        torch.manual_seed(42)
        x = torch.rand(batch_size, dim_l)
        y = torch.rand(batch_size, dim_l)
        scorer = ScaledDotProduct()
        score = torch.sum(scorer.score(x, y)).item()
        self.assertAlmostEqual(score, 4.8168, places=4)

    def test_cosine_similarity(self):
        batch_size = 2
        dim_l = 32
        torch.manual_seed(42)
        x = torch.rand(batch_size, dim_l)
        y = torch.rand(batch_size, dim_l)
        scorer = CosineSimilarity()
        score = torch.sum(scorer.score(x, y)).item()
        self.assertAlmostEqual(score, 2.8963, places=4)


if __name__ == '__main__':
    unittest.main()

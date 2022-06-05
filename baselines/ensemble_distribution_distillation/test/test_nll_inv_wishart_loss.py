import torch
import unittest
import src.loss as loss
import numpy as np


class TestWishartLoss(unittest.TestCase):
    def test_wishart_nll_one_dim(self):
        B, N, D = 1, 1, 1
        target = torch.tensor([[3.5]], dtype=torch.float).reshape((B, N, D))
        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        inv_wish_nll = loss.inv_wish_nll((psi, nu), target)
        self.assertAlmostEqual(inv_wish_nll.item(),
                               np.math.lgamma(5) + 1.982816,
                               places=5)

    def test_wishart_nll_two_dim(self):
        B, N, D = 1, 1, 2
        target = torch.tensor([[3.5, 2.0]], dtype=torch.float).reshape((B, N, D))
        psi = torch.tensor([[4.0, 2.5]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        inv_wish_nll = loss.inv_wish_nll((psi, nu), target)
        self.assertAlmostEqual(inv_wish_nll.item(),
                               np.math.lgamma(5) + 3.06673,
                               places=5)

    def test_wishart_nll_two_ensemble_members(self):
        B, N, D = 1, 2, 1
        target = torch.tensor([[3.5, 2.5]], dtype=torch.float).reshape((B, N, D))
        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10.0]], dtype=torch.float)

        inv_wish_nll = loss.inv_wish_nll((psi, nu), target)
        self.assertAlmostEqual(inv_wish_nll.item(),
                               np.math.lgamma(5) + 1.24737,
                               places=5)

    def test_wishart_nll_batch(self):
        B, N, D = 2, 1, 1
        target = torch.tensor([[3.5], [2.5]], dtype=torch.float).reshape((B, N, D))
        psi = torch.tensor([[4.0], [2.0]], dtype=torch.float)
        nu = torch.tensor([[10.0], [5.0]], dtype=torch.float)

        inv_wish_nll = loss.inv_wish_nll((psi, nu), target)
        self.assertAlmostEqual(inv_wish_nll.item(),
                               0.5 * np.math.lgamma(5) +
                               0.5 * np.math.lgamma(2.5) + 1.991126,
                               places=5)


if __name__ == '__main__':
    unittest.main()

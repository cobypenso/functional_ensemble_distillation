import torch
import unittest
import math
import src.loss as loss
import math

NUM_DECIMALS = 5


class TestGaussianNll(unittest.TestCase):
    def test_one_dim(self):
        B, N, D = 1, 2, 1
        target = torch.tensor([[1.0, -2.0]],
                              dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.0]], dtype=torch.float)
        var = torch.tensor([[1.0]], dtype=torch.float)

        true_sample_avg = -1 / 2
        true_sample_cov = 9 / 4

        gauss_nll = loss.gaussian_nll_1d((mean, var), target)
        true_nll = (math.log(2 * math.pi) + math.log(1) +
                    ((true_sample_avg - 0)**2 + true_sample_cov) / 1) / 2
        self.assertAlmostEqual(gauss_nll.item(), true_nll, places=NUM_DECIMALS)

    def test_two_dim(self):
        B, N, D = 1, 1, 2
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.5, 1.0]], dtype=torch.float)
        var = torch.tensor([[2.0, 5.0]], dtype=torch.float)

        gauss_nll = loss.gaussian_neg_log_likelihood((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               math.log(2 * math.pi) + 0.5 * math.log(10) +
                               0.1625,
                               places=NUM_DECIMALS)

    def test_two_ensemble_members(self):
        B, N, D = 1, 2, 1
        target = torch.unsqueeze(torch.tensor([[1.0, 0.75]],
                                              dtype=torch.float),
                                 dim=-1).reshape(B, N, D)
        mean = torch.tensor([[0.5]], dtype=torch.float)
        var = torch.unsqueeze(torch.tensor([10.0], dtype=torch.float), dim=-1)

        gauss_nll = loss.gaussian_nll_1d((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               0.25 * math.log(4 * math.pi**2 * 10**2) +
                               (0.015625 / 2),
                               places=NUM_DECIMALS)

    def test_batch(self):
        B, N, D = 2, 1, 1
        target = torch.tensor([[1.0], [0.75]],
                              dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.5], [0.25]], dtype=torch.float)
        var = torch.tensor([[10.0], [5.0]], dtype=torch.float)

        gauss_nll = loss.gaussian_nll_1d((mean, var), target)
        true_nll_1 = (math.log(2 * math.pi) + math.log(10.0) +
                      ((1.0 - 0.5)**2 + 0) / 10.0) / 2
        true_nll_2 = (math.log(2 * math.pi) + math.log(5.0) +
                      ((0.75 - 0.25)**2 + 0) / 5.0) / 2
        self.assertAlmostEqual(gauss_nll.item(), (true_nll_1 + true_nll_2) / 2,
                               places=NUM_DECIMALS)


if __name__ == '__main__':
    unittest.main()

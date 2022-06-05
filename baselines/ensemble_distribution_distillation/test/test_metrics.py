import unittest
import torch
import torch_testing as tt
from src import utils
from src import metrics

NUM_DECIMALS = 5


class TestMetrics(unittest.TestCase):
    def test_entropy(self):
        predictions = torch.tensor([0.3, 0.7])
        entropy = metrics.entropy(predictions, None)
        self.assertAlmostEqual(entropy.item(), 0.61086430205)

    def test_entropy_batch(self):
        predictions = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        entropy = metrics.entropy(predictions, None)
        tt.assert_almost_equal(entropy,
                               torch.tensor([0.61086430205, 0.61086430205]))

    def test_accuracy(self):
        true_label = torch.tensor(0).int()
        predictions = torch.tensor([0.9, 0.1])
        acc = metrics.accuracy(predictions, true_label.long())
        self.assertAlmostEqual(acc, 1)

    def test_accuracy_batch(self):
        true_label = torch.tensor([1, 0, 2, 0]).int()
        predictions = torch.tensor([[0.05, 0.09, 0.05], [0.1, 0.8, 0.1],
                                    [0.1, 0.2, 0.7], [0.25, 0.5, 0.25]])
        acc = metrics.accuracy(predictions, true_label.long())
        self.assertAlmostEqual(acc, 0.5)

    def test_squared_error(self):
        targets = torch.tensor([1, 2, 1.5])
        predictions = torch.tensor([[0.9, 2.1, 1.7, 0.0, 0.0, 0.0]])
        squared_error = metrics.squared_error(predictions, targets)
        self.assertAlmostEqual(squared_error, 0.02)

    def test_mean_squared_error(self):
        B, N, D = 1, 3, 1
        targets = torch.tensor([0.9, 1, 1.1]).reshape((B, N, D))
        regression_estimate = torch.tensor([1]).reshape((B, D))
        mse = metrics.mean_squared_error(regression_estimate.float(), targets)
        self.assertAlmostEqual(mse.item(), (0.02 / 3))

    def test_root_mean_squared_error(self):
        B, N, D = 2, 1, 1
        targets = torch.tensor([6, 6]).reshape((B, N, D))
        regression_estimate = torch.tensor([3.0402, 2.6091]).reshape((B, D))
        rmse = metrics.root_mean_squared_error(regression_estimate,
                                               targets.float())
        self.assertAlmostEqual(rmse.item(),
                               3.1826576041101244,
                               places=NUM_DECIMALS)

    def test_uncertainty_separation_entropy(self):
        predictions = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
        pred_mean = torch.mean(predictions, dim=1)
        self.assertAlmostEqual(torch.sum(pred_mean), 1.0)
        tot_unc, ep_unc, al_unc = metrics.uncertainty_separation_entropy(
            predictions, None)
        self.assertAlmostEqual(tot_unc.item(), 0.8813, places=4)
        self.assertAlmostEqual(ep_unc.item(), 0.0349, places=4)
        self.assertAlmostEqual(al_unc.item(), 0.8464, places=4)


if __name__ == '__main__':
    unittest.main()

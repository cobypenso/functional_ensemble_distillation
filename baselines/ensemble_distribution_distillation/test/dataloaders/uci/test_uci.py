"""Test: UCI Dataloader"""
import unittest
import numpy as np
import src.dataloaders.uci.wine as uci_wine
import src.dataloaders.uci.uci_base as uci_base
import src.dataloaders.uci.bost as uci_bost

NUM_DECIMALS = 5


class TestWineData(unittest.TestCase):
    def setUp(self):
        base = uci_wine.WineData(
            "~/doktor/datasets/UCI/wine/winequality-red.csv")
        self.wine_data = uci_base._UCIDataset(base.data[:, :-1],
                                              base.data[:, -1:])

    def test_num_samples(self):
        self.assertEqual(len(self.wine_data), 1599)
        self.assertEqual(len(self.wine_data), self.wine_data.num_samples)

    def test_dim(self):
        input_, target = self.wine_data[0]
        self.assertEqual(input_.shape[0], 11)
        self.assertIsInstance(target[0], float)


class TestBostonData(unittest.TestCase):
    def setUp(self):
        base = uci_bost.BostonData("~/doktor/datasets/UCI/bost/housing.data")
        self.bost_data = uci_base._UCIDataset(base.data[:, :-1],
                                              base.data[:, -1:])

    def test_num_samples(self):
        self.assertEqual(len(self.bost_data), 506)
        self.assertEqual(len(self.bost_data), self.bost_data.num_samples)

    def test_dim(self):
        input_, target = self.bost_data[0]
        self.assertEqual(input_.shape[0], 13)
        self.assertIsInstance(target[0], float)

    def test_first_row(self):
        input_, target = self.bost_data[0]
        self.assertAlmostEqual(input_[0].item(), 0.00632, places=NUM_DECIMALS)
        self.assertAlmostEqual(target, 24.00, places=NUM_DECIMALS)


if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from src import utils
import torch_testing as tt


class TestMetrics(unittest.TestCase):
    def test_to_one_hot(self):
        label = torch.tensor(1)
        num_classes = 2
        one_hot = utils.to_one_hot(label, num_classes)
        self.assertEqual(one_hot.shape[0], num_classes)
        self.assertEqual(one_hot[0], 0)
        self.assertEqual(one_hot[1], 1)

    def test_argmax_1d(self):
        input_tensor = torch.tensor([1, 0])
        ind, value = utils.tensor_argmax(input_tensor)
        self.assertEqual(ind, 0)
        self.assertEqual(value, 1)

    def test_argmax_2d(self):
        input_tensor = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
        ind, value = utils.tensor_argmax(input_tensor)
        tt.assert_equal(ind, torch.tensor([0, 1]))
        tt.assert_equal(value, torch.tensor([0.9, 0.7]))


if __name__ == '__main__':
    unittest.main()

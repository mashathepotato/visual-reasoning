from __future__ import annotations

import unittest

import torch

from utils.fot.supervised_models import PairCNN, PairVisionTransformer, count_parameters


class SupervisedModelTests(unittest.TestCase):
    def test_pair_cnn_shape_and_parameters(self) -> None:
        model = PairCNN()
        output = model(torch.zeros((2, 6, 64, 64)))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertGreater(count_parameters(model, trainable_only=True), 0)

    def test_pair_vit_shape_and_validation(self) -> None:
        model = PairVisionTransformer(embed_dim=32, depth=1, num_heads=4, patch_size=16)
        output = model(torch.zeros((2, 6, 64, 64)))
        self.assertEqual(tuple(output.shape), (2, 2))
        with self.assertRaises(ValueError):
            model(torch.zeros((2, 3, 64, 64)))


if __name__ == "__main__":
    unittest.main()

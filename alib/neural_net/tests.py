__author__ = 'Alok'

import unittest
import numpy as np

from alib.neural_net import NeuralNet, NeuralNetError


class NeuralNetTestCase(unittest.TestCase):
    invalid_layers = [
        [],
        [0],
        [3],
        [-4, 3],
        [-3, 0],
        [5, -2],
        [2, 0],
        [0, 2],
        [0, 0],
        [-2, -5],
    ]
    valid_layers = [
        [1, 2],
        [10, 5],
        [100, 12, 10],
        [1000, 30, 1],
        [40, 42, 12, 4]
    ]
    weights_for_valid_layers = [
        [(2, 2)],
        [(5, 11)],
        [(12, 101), (10, 13)],
        [(30, 1001), (1, 31)],
        [(42, 41), (12, 43), (4, 13)]
    ]

    def setUp(self):
        pass

    def test_instance_creation(self):
        for layers in self.valid_layers:
            self.assertIsInstance(NeuralNet(layers), NeuralNet)

        for layers in self.invalid_layers:
            with self.assertRaises(NeuralNetError):
                NeuralNet(layers)

    def test_random_initialize_weights(self):
        for i, layers in enumerate(self.valid_layers):
            net = NeuralNet(layers)
            net.random_initialize_weights()

            # check shape of weights
            self.assertEqual([weight.shape for weight in net.weights], self.weights_for_valid_layers[i])

if __name__ == "__main__":
    unittest.main()

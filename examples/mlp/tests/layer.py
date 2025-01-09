import unittest
import numpy as np
from activations import ReLU, Softmax, Activation
from layers import Flatten, Dense
from optimizers import SGD


class TestLayers(unittest.TestCase):

    # Tests for Flatten Layer

    def test_flatten_large_input(self):
        """Test Flatten with large input"""
        flatten = Flatten(shape=(100, 100))
        X = np.random.rand(1, 100, 100)
        output = flatten.run(X)
        self.assertEqual(output.size, 10000)
        self.assertEqual(output.shape, (X.shape[0], 10000))

    def test_flatten_invalid_shape(self):
        """Test Flatten with invalid shape to see if it throws errors"""
        flatten = Flatten(shape=(4, 4))
        with self.assertRaises(AssertionError):
            flatten.run(np.array([[1, 2, 3]]))  # Invalid input size

    def test_flatten_update_no_effect(self):
        """Test that update_parameters in Flatten has no effect"""
        flatten = Flatten(shape=(4, 4))
        gradient = np.array([1, 2, 3, 4])
        updated_gradient = flatten.update_parameters(gradient, lr=0.01, optimizer=SGD())
        np.testing.assert_array_equal(updated_gradient, gradient)

    # Tests for Dense Layer

    def test_dense_initialization(self):
        """Test Dense layer weight and bias initialization"""
        shape = (3, 5)
        dense = Dense(shape=shape, activation=ReLU)
        self.assertEqual(dense.weights.shape, (5, 3))
        self.assertEqual(dense.biases.shape, (1, 5))
        self.assertTrue(np.all(dense.weights < 1))
        self.assertTrue(np.all(dense.biases < 1))

        # def test_dense_run_invalid_input(self):
        """Test Dense layer with invalid input size"""
        shape = (3, 4)  # Expecting input size of 3
        dense = Dense(shape=shape, activation=ReLU)
        invalid_input = np.array([[1, 2, 3, 4]])  # Wrong size
        with self.assertRaises(AssertionError):
            dense.run(invalid_input)

    def test_dense_no_activation(self):
        """Test Dense layer without any activation"""
        shape = (2, 3)
        dense = Dense(shape=shape, activation=ReLU)
        dense.weights = np.array([[1, 2], [1, 2], [1, 2]])
        dense.biases = np.array([1, 1, 1])
        X = np.array([[1, 1]])
        output = dense.run(X)
        expected_output = np.array([[4, 4, 4]])
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_dense_backpropagation_no_activation(self):
        """Test backpropagation for Dense layer without activation gradient"""
        shape = (3, 4)
        dense = Dense(shape=shape, activation=ReLU)
        X = np.array([[1, 2, 3]])
        dense.run(X)  # Forward pass to store Z, A, X
        gradient = np.array([0.1, 0.2, 0.3, 0.4])  # Mock gradient
        lr = 0.01
        optimizer = SGD()
        updated_gradient = dense.update_parameters(gradient, lr, optimizer)

        # Check the update process
        da = np.ones_like(dense.Z) * gradient  # Identity gradient
        dw = np.outer(da, X)
        db = np.sum(da, keepdims=True)
        new_weights, new_biases = optimizer.update(dense.weights, dense.biases, dw, db)
        np.testing.assert_array_almost_equal(dense.weights, new_weights)
        np.testing.assert_array_almost_equal(dense.biases, new_biases)

    def test_dense_update_invalid_gradient(self):
        """Test Dense layer's update with invalid gradient size"""
        shape = (3, 4)
        dense = Dense(shape=shape, activation=ReLU)
        X = np.array([[1, 2, 3]])
        dense.run(X)  # Forward pass
        invalid_gradient = np.array([1, 1])  # Wrong size gradient
        with self.assertRaises(ValueError):
            dense.update_parameters(invalid_gradient, lr=0.01, optimizer=SGD())


if __name__ == "__main__":
    unittest.main()

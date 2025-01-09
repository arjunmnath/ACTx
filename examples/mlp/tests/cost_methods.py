import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from costs import SparseCategoricalCE


class TestSparseCategoricalCrossentropy(unittest.TestCase):
    def setUp(self):
        """
        Set up test cases that will be used across multiple tests
        """
        # Simple case with perfect predictions
        self.y_true_perfect = np.array([0, 1, 2])
        self.y_pred_perfect = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # Realistic case with non-perfect predictions
        self.y_true_realistic = np.array([1, 0, 2])
        self.y_pred_realistic = np.array(
            [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]
        )

        # Edge case with very small probabilities
        self.y_true_edge = np.array([0])
        self.y_pred_edge = np.array([[0.0001, 0.9998, 0.0001]])
        self.scce = SparseCategoricalCE()

    def test_perfect_predictions(self):
        """
        Test loss and gradient for perfect predictions (should be close to zero)
        """
        loss = self.scce.cost(self.y_true_perfect, self.y_pred_perfect)
        assert_almost_equal(loss, 0.0, decimal=5)

    def test_realistic_predictions(self):
        """
        Test loss and gradient for realistic predictions
        """
        # Test loss
        loss = self.scce.cost(self.y_true_realistic, self.y_pred_realistic)
        # Expected loss calculated by hand
        expected_loss = -np.mean([np.log(0.7), np.log(0.8), np.log(0.7)])
        assert_almost_equal(loss, expected_loss, decimal=5)

    def test_gradient_shape(self):
        """
        Test that gradient has correct shape
        """
        gradient = self.scce.gradient(self.y_true_realistic, self.y_pred_realistic)
        self.assertEqual(gradient.shape, self.y_pred_realistic.shape)

    def test_gradient_perfect_predictions(self):
        """
        Test gradient values for perfect predictions
        """
        gradient = self.scce.gradient(self.y_true_perfect, self.y_pred_perfect)
        # For perfect predictions, gradient should be -1/batch_size at correct indices
        expected_gradient = np.zeros_like(self.y_pred_perfect)
        batch_size = len(self.y_true_perfect)
        for i, label in enumerate(self.y_true_perfect):
            expected_gradient[i, label] = -1.0 / batch_size
        assert_array_almost_equal(gradient, expected_gradient)

    def test_numerical_stability(self):
        """
        Test that the functions handle very small probabilities without numerical issues
        """
        # Should not raise any numerical warnings or errors
        loss = self.scce.cost(self.y_true_edge, self.y_pred_edge)
        gradient = self.scce.gradient(self.y_true_edge, self.y_pred_edge)

        # Loss should be finite
        self.assertTrue(np.isfinite(loss))
        # Gradient should be finite
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_input_validation(self):
        """
        Test that functions properly handle invalid inputs
        """
        # Test with mismatched dimensions
        invalid_y_pred = np.array([[0.5, 0.5]])
        invalid_y_true = np.array([0, 1])

        with self.assertRaises(IndexError):
            self.scce.cost(invalid_y_true, invalid_y_pred)

    def test_probability_sum(self):
        """
        Test that predictions that sum to 1 give expected results
        """
        y_true = np.array([1])
        y_pred = np.array([[0.2, 0.5, 0.3]])  # Sums to 1

        loss = self.scce.cost(y_true, y_pred)
        expected_loss = -np.log(0.5)  # -log of probability for correct class
        assert_almost_equal(loss, expected_loss, decimal=5)

    def test_batch_independence(self):
        """
        Test that each sample in the batch is processed independently
        """
        y_true = np.array([0, 1])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])

        # Calculate loss for full batch
        full_batch_loss = self.scce.cost(y_true, y_pred)

        # Calculate loss for each sample separately and average
        loss1 = self.scce.cost(np.array([0]), y_pred[0:1])
        loss2 = self.scce.cost(np.array([1]), y_pred[1:2])
        individual_avg_loss = (loss1 + loss2) / 2

        assert_almost_equal(full_batch_loss, individual_avg_loss, decimal=5)


if __name__ == "__main__":
    unittest.main()

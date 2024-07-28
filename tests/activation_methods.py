import unittest
import numpy as np
from activations import ReLU, softmax, get_derivative

ReLU_derivative = get_derivative(ReLU)
softmax_derivative = get_derivative(softmax)


class TestReLUFunction(unittest.TestCase):

    def test_ReLU_positive_values(self):
        np.testing.assert_array_equal(ReLU(np.array([1, 2, 3])), np.array([1, 2, 3]))

    def test_ReLU_negative_values(self):
        np.testing.assert_array_equal(ReLU(np.array([-1, -2, -3])), np.array([0, 0, 0]))

    def test_ReLU_mixed_values(self):
        np.testing.assert_array_equal(ReLU(np.array([-1, 0, 1, 2])), np.array([0, 0, 1, 2]))

    def test_ReLU_zeros(self):
        np.testing.assert_array_equal(ReLU(np.array([0, 0, 0])), np.array([0, 0, 0]))


class TestSoftmaxFunction(unittest.TestCase):

    def test_softmax_standard_case(self):
        input_array = np.array([1.0, 2.0, 3.0])
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        np.testing.assert_array_almost_equal(softmax(input_array), expected_output, decimal=5)

    def test_softmax_all_zeros(self):
        input_array = np.array([0.0, 0.0])
        expected_output = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(softmax(input_array), expected_output, decimal=5)

    def test_softmax_large_values(self):
        input_array = np.array([1000, 1000])
        expected_output = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(softmax(input_array), expected_output, decimal=5)

    def test_softmax_mixed_values(self):
        input_array = np.array([1.0, 2.0, 3.0, 4.0])
        expected_output = np.array([0.03205860, 0.08714432, 0.23688282, 0.64391426])
        np.testing.assert_array_almost_equal(softmax(input_array), expected_output, decimal=5)

    def test_relu_derivative_positive_values(self):
        np.testing.assert_array_equal(ReLU_derivative(np.array([1, 2, 3])), np.array([1, 1, 1]))

    def test_relu_derivative_negative_values(self):
        np.testing.assert_array_equal(ReLU_derivative(np.array([-1, -2, -3])), np.array([0, 0, 0]))

    def test_relu_derivative_mixed_values(self):
        np.testing.assert_array_equal(ReLU_derivative(np.array([-1, 0, 1, 2])), np.array([0, 0, 1, 1]))

    def test_relu_derivative_zeros(self):
        np.testing.assert_array_equal(ReLU_derivative(np.array([0, 0, 0])), np.array([0, 0, 0]))


class TestSoftmaxDerivative(unittest.TestCase):

    def test_basic_case(self):
        z = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)  # Column vector
        softmax_output = softmax(z)
        expected_jacobian = np.array([
            [0.01821127, -0.01322146, -0.00498981],
            [-0.01322146, 0.07007415, -0.05685268],
            [-0.00498981, -0.05685268, 0.06184249]
        ])
        np.testing.assert_array_almost_equal(softmax_derivative(softmax_output), expected_jacobian, decimal=5)

    def test_all_zeros(self):
        z = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)  # Column vector
        softmax_output = softmax(z)
        expected_jacobian = np.array([
            [0.33333333, -0.11111111, -0.11111111],
            [-0.11111111, 0.33333333, -0.11111111],
            [-0.11111111, -0.11111111, 0.33333333]
        ])
        np.testing.assert_array_almost_equal(softmax_derivative(softmax_output), expected_jacobian, decimal=5)

    def test_negative_values(self):
        z = np.array([-1.0, -2.0, -3.0]).reshape(-1, 1)  # Column vector
        softmax_output = softmax(z)
        expected_jacobian = np.array([
            [0.09003057, -0.04501529, -0.04501529],
            [-0.04501529, 0.09003057, -0.04501529],
            [-0.04501529, -0.04501529, 0.09003057]
        ])
        np.testing.assert_array_almost_equal(softmax_derivative(softmax_output), expected_jacobian, decimal=5)

    def test_equal_values(self):
        z = np.array([2.0, 2.0, 2.0]).reshape(-1, 1)  # Column vector
        softmax_output = softmax(z)
        expected_jacobian = np.array([
            [0.33333333, -0.11111111, -0.11111111],
            [-0.11111111, 0.33333333, -0.11111111],
            [-0.11111111, -0.11111111, 0.33333333]
        ])
        np.testing.assert_array_almost_equal(softmax_derivative(softmax_output), expected_jacobian, decimal=5)

    def test_large_values(self):
        z = np.array([1000.0, 1000.0, 1000.0]).reshape(-1, 1)  # Column vector
        softmax_output = softmax(z)
        expected_jacobian = np.array([
            [0.33333333, -0.11111111, -0.11111111],
            [-0.11111111, 0.33333333, -0.11111111],
            [-0.11111111, -0.11111111, 0.33333333]
        ])
        np.testing.assert_array_almost_equal(softmax_derivative(softmax_output), expected_jacobian, decimal=5)


if __name__ == '__main__':
    unittest.main()

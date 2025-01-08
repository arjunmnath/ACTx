from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, size, optimizer=None, activation=None):
        self.size = size
        self.shape = None
        self.activation = activation
        self.optimizer = optimizer

    @abstractmethod
    def run(self, X):
        pass

    @abstractmethod
    def update_parameters(self, gradient, lr, l2reg):
        pass


class Flatten(Layer):
    def __init__(self, shape, optimizer, activation=None):
        super().__init__(shape[0], optimizer)
        self.shape = shape[1]

    def run(self, X):
        # assert X.shape == self.shape
        assert (
            X.ndim == 3 and X.shape[1:] == self.shape
        ), f"Shape {X.shape} is not matching {self.shape}, input dimension: {X.ndim}"
        return X.reshape(X.shape[0], -1)

    def init(self, shape=None):
        self.shape = self.size
        self.size = self.shape[0] * self.shape[1]

    def update_parameters(self, gradient, lr, l2reg):
        return gradient


class Dense(Layer):
    def __init__(self, shape, optimizer, activation):
        super().__init__(shape[1], optimizer, activation())
        self.biases = None  # shape: (1, no_of_neurons)
        self.weights = None  # shape: (no_of_neurons, no_of_inputs)
        self.X = None
        self.A = None
        self.Z = None
        self.dtype = np.float32
        self.batch_size = 0
        self.shape = shape
        # Xavier initialization
        self.weights = (
            np.random.rand(*self.shape) * np.sqrt(2.0 / self.shape[1])
        ).astype(self.dtype)
        self.biases = (
            np.random.rand(1, self.shape[0]) * np.sqrt(2.0 / self.shape[1])
        ).astype(self.dtype)

    def __batch_norm(self, Z, epsilon=1e-8):
        """
        :param Z: shape(batch_size, no_of_neurons)
        :param epsilon: small positive value to avoid division by zero
        :return: normalized Z
        """
        mean = np.mean(Z, axis=0, keepdims=True)
        variance = np.var(Z, axis=0, keepdims=True)
        return (Z - mean) / np.sqrt(variance + epsilon)

    def run(self, X):
        """
        :param X: shape(batch_size, no_of_inputs)
        :return: shape(batch_size, no_of_neurons) Activation Values from current layer
        """
        # assert X.shape == (1, self.shape[0])
        assert X.ndim == 2, "Input must be 2-D Array"
        assert (
            X.shape[1] == self.shape[1]
        ), f"Malformed input shape X:{X.shape}, layer:{self.shape}"
        self.batch_size = X.shape[0]
        # Z = self.weights.dot(X).reshape(X.shape[0], -1) + self.biases
        Z = X @ self.weights.T + self.biases
        Z = self.__batch_norm(Z)
        A = self.activation.activate(Z)
        self.Z, self.A, self.X = Z, A, X
        return A

    def update_parameters(self, gradient: np.ndarray, lr, l2reg=0.0001):
        gradient = gradient * self.activation.activate(self.Z)
        dw = gradient.T.dot(self.X) / self.batch_size
        # dw = dw.T @ self.X + l2reg * self.weights
        db = np.sum(gradient, axis=0, keepdims=True) / self.batch_size
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, dw, db
        )
        return np.dot(gradient, self.weights)
        # return da

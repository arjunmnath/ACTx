import numpy as np

from abc import ABC, abstractmethod as abstract


class Activation(ABC):
    @abstract
    def activate(self, x):
        pass

    @abstract
    def gradient(self, x):
        pass


class Sigmoid(Activation):
    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        return x * (1.0 - x)


class ReLU(Activation):
    def activate(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)


class dummy(Activation):
    def activate(self, x):
        return x

    def gradient(self, x):
        return np.where(x > 0, 1, 0)


class Softmax(Activation):
    def activate(self, x):
        assert x.ndim == 2
        z_max = np.max(x, axis=1, keepdims=True)
        exp_z = np.exp(x - z_max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def gradient(self, x):
        # s = self.activate(x)
        # jacobian_matrix = np.diagflat(s) - np.outer(s, s)
        # gradient = np.dot(x, jacobian_matrix)
        # return gradient.reshape((1,) + gradient.shape)
        return self.activate(x)

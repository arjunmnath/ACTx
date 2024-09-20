import numpy as np

from abc import ABC, abstractmethod as abstract

class Activation(ABC):
    @abstract
    def activate(self, x):
        pass

    @abstract
    def gradient(self, x):
        pass

class ReLU(Activation):
    def activate(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(Activation):
    def activate(self, x):
        e = np.exp(x - np.max(x)) # x - np.max() for normalization
        s = np.sum(e)
        return e / s

    def gradient(self, x):
        s = self.activate(x)
        jacobian_matrix = np.diagflat(s) - np.outer(s, s)
        return np.dot(x, jacobian_matrix).reshape(1, -1)

import numpy as np

from abc import ABC, abstractmethod as abstract

class Costs(ABC):
    @abstract
    def cost(self, y_hat, y):
        pass

    @abstract
    def gradient(self, y_hat, y):
        pass

class MSE(Costs):

    def cost(self, y_hat, y):
        return np.square(y_hat - y) / y.shape[1]

    def gradient(self, y_hat, y):
        return (y - y_hat) * 2 / y.shape[1]


class CrossEntropy(Costs):
    def cost(self, y_hat, y):
        assert 0 <= y_hat.all() <= 1 and 0 <= y.all() <= 1
        m = y_hat.shape[0]
        epsilon = 1e-10
        return -(1 / m) * np.sum(y_hat * np.log(y + epsilon) + (1 - y_hat + epsilon) * np.log(1 - y + epsilon))
        return y_hat

    def gradient(self, y_hat, y):
        # Gradient of cross-entropy loss with respect to predictions
        return - (y_hat / (y + 1e-10)) + (1 - y_hat) / (1 - y + 1e-10)

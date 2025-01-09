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
        return np.square(y_hat - y) / y.shape[0]

    def gradient(self, y_hat, y):
        return (y - y_hat) * 2 / y.shape[0]


class CrossEntropy(Costs):
    def cost(self, y_hat, y):
        assert 0 <= y_hat.all() <= 1 and 0 <= y.all() <= 1
        m = y_hat.shape[0]
        epsilon = 1e-10
        return -(1 / m) * np.sum(
            y_hat * np.log(y + epsilon)
            + (1 - y_hat + epsilon) * np.log(1 - y + epsilon)
        )

    def gradient(self, y_hat, y):
        return -(y_hat / (y + 1e-10)) + (1 - y_hat) / (1 - y + 1e-10)


class SparseCategoricalCE(Costs):
    def cost(self, y_true, y_pred, epsilon=1e-6):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def gradient(self, y_true, y_pred, epsilon=1e-6):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        gradient = -y_true / y_pred
        gradient /= y_true.shape[0]
        return gradient

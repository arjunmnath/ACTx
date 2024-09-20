import numpy as np

from abc import ABC, abstractmethod as abstract

class Costs(ABC):
    @abstract
    def cost(self, x_hat, x):
        pass

    @abstract
    def gradient(self, x_hat, x):
        pass

class MSE(Costs):

    def cost(self, x_hat, x):
        return np.average(np.square(x_hat - x))

    def gradient(self, x_hat, x):
        return x_hat - x
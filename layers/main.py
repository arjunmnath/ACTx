import numpy as np
class Layer:
    activation = None
    shape = None
    size = None

    def __init__(self, shape, activation = None):
        self.shape = shape
        self.activation = activation

class Flatten(Layer):
    def __init__(self, shape):
        super().__init__(shape)
        self.size = shape[0] * shape[1]
    def run(self, X):
        return X.flatten()

class Dense(Layer):
    activation = None
    weights = None
    biases = None
    def __init__(self, shape, activation):
        super().__init__(shape, activation)
        self.size = shape[0]
        self.weights = np.random.uniform(low=-2.4/shape[1], high=2.4/shape[1], size=shape)
        self.biases = np.random.uniform(low=-2.4/shape[1], high=2.4/shape[1], size=(self.size,))
    def run(self, X):
        Z_L = np.dot(self.weights, X.T) + self.biases
        A_L = self.activation(Z_L)
        return [Z_L, A_L]

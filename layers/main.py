import numpy as np
from abc import ABC, abstractmethod



class Layer(ABC):
    def __init__(self, shape, activation = None):
        self.shape = shape
        self.activation = activation
    @abstractmethod
    def run(self, X):
        pass

    @abstractmethod
    def update_parameters(self, gradient, lr, optimizer, l2reg):
        pass

class Flatten(Layer):
    def __init__(self, shape):
        super().__init__(shape)
        self.size = shape[0] * shape[1]
    def run(self, X):
        # assert X.shape == self.shape
        assert X.ndim == 3 and X.shape[1:] == self.shape, f"Shape {X.shape} is not matching {self.shape}, input dimension: {X.ndim}"
        return X.reshape(X.shape[0], -1)

    def update_parameters(self, gradient, lr, optimizer, l2reg):
        return gradient

class Dense(Layer):
    def __init__(self, shape, activation):
        super().__init__(shape, activation())
        # keep lambda_reg in range 1e−4 to 1e-2
        self.X = None
        self.A = None
        self.Z = None
        self.size = shape[1]
        # Xavier initialization
        self.weights = np.random.rand(shape[1], shape[0]) * np.sqrt(2. / shape[1])
        self.biases= np.random.rand(1, shape[1]) * np.sqrt(2. / shape[1])




    def run(self, X):
        # assert X.shape == (1, self.shape[0])
        assert X.ndim == 2 and X.shape[1] == self.shape[0]
        Z = self.weights.dot(X.T).reshape(X.shape[0], -1) + self.biases
        A = self.activation.activate(Z)
        self.Z, self.A, self.X = Z, A, X
        return A

    def update_parameters(self, gradient: np.ndarray, lr, optimizer, l2reg):
        da = self.activation.gradient(self.Z) * gradient
        da = np.clip(da, -1, 1)
        dw = da.T @ self.X
        # dw = da.T @ self.X + l2reg * self.weights
        db = np.sum(da, axis=0, keepdims=True)
        self.weights, self.biases = optimizer.update(self.weights, self.biases, dw, db)
        return da.dot(self.weights)
        # return da
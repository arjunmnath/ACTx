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

    def update_parameters(self, gradient):
        return gradient

class Dense(Layer):
    weights = None
    biases = None
    def __init__(self, shape, activation):
        super().__init__(shape, activation())
        self.X = None
        self.A = None
        self.Z = None
        self.size = shape[0]
        self.weights = np.random.uniform(low=-2.4/shape[1], high=2.4/shape[1], size=(shape))
        self.biases = np.random.uniform(low=-2.4/shape[1], high=2.4/shape[1], size=(self.size,))
    def run(self, X):
        assert (len(X.shape) == 1), "Malformed activation values"
        Z = np.dot(self.weights, X.T) + self.biases
        A = self.activation.activate(Z)
        self.Z, self.A, self.X = Z, A, X
        return A

    def update_parameters(self, gradient: np.ndarray, lr):
        # db = gradient * activation_gradient(z)
        # dw = db * X
        # (1, 10)
        db = self.activation.gradient(self.Z) * gradient
        dw = db.T * self.X
        self.weights -= lr * dw
        self.biases -= lr * self.biases
        return db @ self.weights
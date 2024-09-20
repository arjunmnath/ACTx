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

    def update_parameters(self, gradient, lr, optimizer):
        return gradient

class Dense(Layer):
    def __init__(self, shape, activation):
        super().__init__(shape, activation())
        self.X = None
        self.A = None
        self.Z = None
        self.size = shape[0]
        # Xavier initialization
        self.weights = np.random.rand(shape[0], shape[1]) * np.sqrt(2. / shape[1])
        self.biases= np.random.rand(self.size) * np.sqrt(2. / shape[1])

    def run(self, X):
        assert (len(X.shape) == 1), "Malformed activation values"
        Z = np.dot(self.weights, X.T) + self.biases
        A = self.activation.activate(Z)
        self.Z, self.A, self.X = Z, A, X
        return A

    def update_parameters(self, gradient: np.ndarray, lr, optimizer):
        da = self.activation.gradient(self.Z) * gradient
        dw = np.outer(da, self.X)
        db = np.sum(da, axis=1, keepdims=True)
        self.weights, self.biases = optimizer.update(self.weights, self.biases, dw, db)
        return da.dot(self.weights)
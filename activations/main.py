import numpy as np


def ReLU(X):
    return np.maximum(0, X)

def softmax(X):
    e = np.exp(X - np.max(X))
    s = np.sum(e)
    return e/s

def ReLU_derivative(X):
    return np.where(X > 0, 1, 0)

def softmax_derivative(X):

    s = softmax(X).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def get_derivative(activation):
    if callable(activation):
        if activation.__name__ == "ReLU":
            return ReLU_derivative
        elif activation.__name__ == "softmax":
            return softmax_derivative
    else:
        raise ValueError("Non Callable activation received...")

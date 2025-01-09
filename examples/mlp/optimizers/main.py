import numpy as np


class Optimizer:
    def update(self, weights, biases, dw, db):
        raise NotImplementedError("Update method not implemented")


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_w, self.v_w = None, None
        self.m_b, self.v_b = None, None
        self.t = 0

    def initialize(self, weights_shape, biases_shape):
        self.m_w = np.zeros(weights_shape)
        self.v_w = np.zeros(weights_shape)
        self.m_b = np.zeros(biases_shape)
        self.v_b = np.zeros(biases_shape)

    def update(self, weights, biases, dw, db):
        if self.m_w is None:
            self.initialize(weights.shape, biases.shape)

        self.t += 1
        # Adam update for weights
        self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * dw
        self.v_w = self.beta_2 * self.v_w + (1 - self.beta_2) * (dw**2)
        m_w_hat = self.m_w / (1 - self.beta_1**self.t)
        v_w_hat = self.v_w / (1 - self.beta_2**self.t)
        weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Adam update for biases
        self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * db
        self.v_b = self.beta_2 * self.v_b + (1 - self.beta_2) * (db**2)
        m_b_hat = self.m_b / (1 - self.beta_1**self.t)
        v_b_hat = self.v_b / (1 - self.beta_2**self.t)
        biases -= (self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)).flatten()

        return weights, biases


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, weights, biases, dw, db):
        weights -= self.lr * dw
        biases -= self.lr * db
        return weights, biases

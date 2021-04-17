import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.cache = None  # for backpropagation
        self.weights = np.sqrt(2 / in_features) * np.random.randn(in_features, out_features)  # He initialization
        self.bias = 1e-2 * np.ones(out_features)
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, activations):
        self.cache = activations
        return np.dot(activations, self.weights) + self.bias.reshape(1, -1)

    def backward(self, gradients):
        self.weights_grad += np.dot(np.transpose(self.cache), gradients)
        self.bias_grad += np.sum(gradients, axis=0)
        return np.dot(gradients, np.transpose(self.weights))

    def step(self, lr):
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

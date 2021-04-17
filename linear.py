import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        """
        :param in_features: number of input neurons
        :param out_features: number of output neurons
        """
        self.cache = None  # for backpropagation
        self.weights = np.sqrt(2 / in_features) * np.random.randn(in_features, out_features)  # He initialization
        self.bias = 1e-2 * np.ones(out_features)
        self.weights_grad = np.zeros_like(self.weights)  # weight gradients
        self.bias_grad = np.zeros_like(self.bias)  # bias gradients

    def __call__(self, activations):
        """
        Forward passes the activations.
        :param activations: input activations
        :return: next layer activations
        """
        self.cache = activations
        return np.dot(activations, self.weights) + self.bias.reshape(1, -1)

    def backward(self, gradients):
        """
        Back propagates the upstream gradients, while accumulating the weight & bias gradients
        :param gradients: upstream gradients
        :return: gradients for the previous layer
        """
        self.weights_grad += np.dot(np.transpose(self.cache), gradients)
        self.bias_grad += np.sum(gradients, axis=0)
        return np.dot(gradients, np.transpose(self.weights))

    def step(self, lr):
        """
        Updates weights & biases
        :param lr: learning rate
        """
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad

    def zero_grad(self):
        """
        Sets the weight & bias gradients to 0
        """
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

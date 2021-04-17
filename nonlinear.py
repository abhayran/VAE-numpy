import numpy as np


class Activation:  # base class, linear activation by default
    def __init__(self):
        self.cache = None

    def __call__(self, activations):
        return activations

    def backward(self, gradients):
        return gradients


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, activations):
        self.cache = np.copy(activations)
        return activations * (activations > 0)

    def backward(self, gradients):
        return gradients * (self.cache > 0)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, activations):
        self.cache = np.copy(activations)
        return 1 / (1 + np.exp(-activations))

    def backward(self, gradients):
        next_ = self(self.cache)
        return gradients * next_ * (1 - next_)

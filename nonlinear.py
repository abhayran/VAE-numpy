import numpy as np


class ReLU:
    def forward_prop(self, input_activations):
        return input_activations * (input_activations > 0)

    def backward_prop(self, input_activations, incoming_gradients):
        return incoming_gradients * (input_activations > 0)


class Sigmoid:
    def forward_prop(self, input_activations):
        pass

    def backward_prop(self, incoming_gradients):
        pass


class Softmax:
    def forward_prop(self, input_activations):
        pass

    def backward_prop(self, incoming_gradients):
        pass

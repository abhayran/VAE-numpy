import numpy as np


class Dense:
    def __init__(self, n_of_prev_nodes, n_of_nodes, learning_rate=1e-4):
        self.n_of_prev_nodes = n_of_prev_nodes  # number of previous nodes
        self.n_of_nodes = n_of_nodes  # number of current nodes
        self.learning_rate = learning_rate  # learning rate
        # Xavier-Glorot weight initialization
        c = np.sqrt(2 / (n_of_nodes + n_of_prev_nodes))
        self.weights = c * np.random.randn(n_of_nodes, n_of_prev_nodes)  # weights connecting this layer to previous layer, each row corresponds to a node
        self.bias = c * np.random.randn(n_of_nodes)

    def forward_prop(self, input_activations):  # implements a forward pass of a batch across two layers
        # input_activations: each column corresponds to the activations of a sample in the batch
        return np.dot(self.weights, input_activations) + self.bias.reshape(-1, 1)

    def backProp(self, incoming_gradients, previous_activations, current_activations):
        batch_size = incoming_gradients.shape[1]
        outgoing_gradients = np.dot(np.transpose(self.weights), incoming_gradients)  # gradients to be passed to the previous layer
        # weights & biases update
        self.weights -= (self.learning_rate / batch_size) * np.dot(incoming_gradients, np.transpose(previous_activations))
        self.bias -= (self.learning_rate / batch_size) * np.sum(incoming_gradients, axis=1)
        return outgoing_gradients  # pass the gradients to the previous layer

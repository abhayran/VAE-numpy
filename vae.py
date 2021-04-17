from linear import Linear
from nonlinear import ReLU, Sigmoid
import numpy as np
import matplotlib.pyplot as plt


class VAE:
    def __init__(self, hidden_sizes, latent_size=2):
        """
        :param hidden_sizes: list, specifying number of neurons for encoder & decoder hidden layers
        :param latent_size: int, latent space dimension
        """
        self.latent_size = latent_size

        # building the encoder
        self.encoder = [Linear(28 ** 2, hidden_sizes[0]), ReLU()]
        for i in range(len(hidden_sizes) - 1):
            self.encoder.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.encoder.append(ReLU())
        self.encoder.append(Linear(hidden_sizes[-1], 2 * latent_size))

        # building the decoder
        self.decoder = [Linear(latent_size, hidden_sizes[-1]), ReLU()]
        for i in range(len(hidden_sizes) - 1):
            self.decoder.append(Linear(hidden_sizes[-(i + 1)], hidden_sizes[-(i + 2)]))
            self.decoder.append(ReLU())
        self.decoder.append(Linear(hidden_sizes[0], 28 ** 2))
        self.decoder.append(Sigmoid())

    def step(self, lr):
        for layer in self.encoder + self.decoder:
            if hasattr(layer, 'step'):
                layer.step(lr)
                layer.zero_grad()

    def train(self, data_loader, epochs, lr):
        for epoch in range(epochs):
            for data, _ in data_loader:
                activations = np.copy(data)

                for layer in self.encoder:
                    activations = layer(activations)
                means, log_covs = activations[:, :self.latent_size], activations[:, self.latent_size:]
                kl_gradients = np.append(means, 0.5 * (np.exp(log_covs) - 1), axis=1)

                gaussian_random = np.random.randn(means.shape[0], means.shape[1])
                activations = means + np.multiply(np.exp(log_covs * 0.5), gaussian_random)

                for layer in self.decoder:
                    activations = layer(activations)
                reconstruction_gradients = 2 * (activations - data)

                for layer in list(reversed(self.decoder)):
                    reconstruction_gradients = layer.backward(reconstruction_gradients)

                gradients = kl_gradients + np.append(reconstruction_gradients, 0.5 * np.multiply(np.multiply(
                    reconstruction_gradients, gaussian_random), np.exp(0.5 * log_covs)), axis=1)

                for layer in list(reversed(self.encoder)):
                    gradients = layer.backward(gradients)

                self.step(lr)

    def visualize_latent_space(self, data, labels):
        activations = np.copy(data)
        for layer in self.encoder:
            activations = layer(activations)
        plt.figure(figsize=(8, 8))
        plt.scatter(activations[:, 0], activations[:, 1], c=labels)
        plt.colorbar()
        plt.show()

    def visualize_manifold(self):
        activations = np.append(np.repeat(np.linspace(-2, 2, 11), 11).reshape((-1, 1)),
                                np.tile(np.linspace(-2, 2, 11), 11).reshape((-1, 1)), axis=1)
        for layer in self.decoder:
            activations = layer(activations)
        figure = np.zeros((308, 308))
        for i in range(121):
            figure[(i // 11) * 28: (i // 11 + 1) * 28, (i % 11) * 28: (i % 11 + 1) * 28] = activations[i, :].reshape((28, 28))
        plt.figure(figsize=(8, 8))
        plt.imshow(figure)
        plt.show()

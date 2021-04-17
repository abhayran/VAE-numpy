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
        """
        Update weights & biases for layers with learnable parameters
        :param lr: learning rate
        """
        for layer in self.encoder + self.decoder:
            if hasattr(layer, 'step'):
                layer.step(lr)
                layer.zero_grad()

    def train(self, data_loader, epochs, lr):
        """
        Trains the variational autoencoder consisting of the layers in <self.encoder> and <self.decoder>
        :param data_loader: data loader, yields NumPy arrays of shape (batch size, 28 ** 2)
        :param epochs: number of epochs
        :param lr: learning rate
        """
        losses = []
        for epoch in range(epochs):
            loss = 0.
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
                loss += np.sum((activations - data) * (activations - data))

                for layer in list(reversed(self.decoder)):
                    reconstruction_gradients = layer.backward(reconstruction_gradients)

                gradients = kl_gradients + np.append(reconstruction_gradients, 0.5 * np.multiply(np.multiply(
                    reconstruction_gradients, gaussian_random), np.exp(0.5 * log_covs)), axis=1)

                for layer in list(reversed(self.encoder)):
                    gradients = layer.backward(gradients)

                self.step(lr)
            losses.append(loss / len(data_loader))

        plt.figure()
        plt.title('Reconstruction loss')
        plt.plot(np.array(list(range(epochs))) + 1, losses)
        plt.show()

    def visualize_latent_space(self, data, labels):
        """
        Forward passes the data through the encoder and visualizes the latent space with respect to the labels
        :param data: NumPy array of shape (number of images, 28 ** 2)
        :param labels: NumPy array of shape (number of images, )
        """
        activations = np.copy(data)
        for layer in self.encoder:
            activations = layer(activations)
        plt.figure(figsize=(12, 12))
        plt.title('Latent space visualization')
        plt.scatter(activations[:, 0], activations[:, 1], c=labels)
        plt.colorbar()
        plt.show()

    def visualize_manifold(self):
        """
        Samples a square grid in the latent space in the range [-2, 2] and visualizes the output from the decoder
        """
        activations = np.append(np.repeat(np.linspace(-2, 2, 11), 11).reshape((-1, 1)),
                                np.tile(np.linspace(-2, 2, 11), 11).reshape((-1, 1)), axis=1)
        for layer in self.decoder:
            activations = layer(activations)
        figure = np.zeros((308, 308))
        for i in range(121):
            figure[(i // 11) * 28: (i // 11 + 1) * 28, (i % 11) * 28: (i % 11 + 1) * 28] = activations[i, :].reshape((28, 28))
        plt.figure(figsize=(12, 12))
        plt.title('Output manifold visualization')
        plt.imshow(figure)
        plt.show()

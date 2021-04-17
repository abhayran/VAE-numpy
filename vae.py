from linear import Linear
from nonlinear import ReLU, Sigmoid
import numpy as np
import matplotlib.pyplot as plt


class VAE:
    def __init__(self):
        self.encoder = []  # will hold the encoder layers
        self.decoder = []  # will hold the decoder layers

    def kl_loss(self, mean, log_cov):  # calculates the KL divergence of N(mean, exp(log_cov)) from N(0, I)
        # log_cov contains the natural logarithm of the diagonal elements in the covariance matrix, rest of the elements are 0.
        return 0.5 * (np.dot(mean, mean) + np.sum(np.exp(log_cov) - log_cov) - mean.shape[0])

    def sampling(self, means, log_covs):  # generates samples for each Gaussian distribution in the batch
        gaussian_random = np.random.randn(means.shape[0], means.shape[1])  # drawing samples from N(0, I)
        samples = means + np.multiply(np.exp(log_covs * 0.5), gaussian_random)  # scaling & shifting the samples in gaussian_random
        return [samples, gaussian_random]

    def visualize_encodings(self, data, labels):  # visualizing the latent space embedding mean vectors
        activations = np.copy(np.transpose(data))  # input activations
        for layer in self.encoder:  # feeding input activations forward through the encoder
            activations = layer.forwardProp(activations)
        # displaying the mean vectors of latent space encoding distributions
        plt.figure(figsize=(8, 8))
        plt.scatter(activations[0, :], activations[1, :], c=labels)
        plt.colorbar()
        plt.show()

    def visualize_manifold(self):
        # taking 121 equally spaced samples from the latent space, encompassing a central square grid having a range (-2, 2) for both axes
        activations = np.append(np.repeat(np.linspace(-2, 2, 11), 11).reshape((1, -1)), np.tile(np.linspace(-2, 2, 11), 11).reshape((1, -1)), axis=0)
        for layer in self.decoder:  # feeding latent activations forward through the decoder
            activations = layer.forwardProp(activations)
        # displaying 121 (28 x 28) images in a square grid structure
        figure = np.zeros((308, 308))
        for i in range(121):
            figure[(i // 11) * 28: (i // 11 + 1) * 28, (i % 11) * 28: (i % 11 + 1) * 28] = activations[:, i].reshape((28, 28))
        plt.figure(figsize=(8, 8))
        plt.imshow(figure)
        plt.show()

    def train(self, training_data, batch_size, epochs, decay, verbose=True):  # trains a variational autoencoder
        batches = (training_data.shape[0] - 1) // batch_size + 1  # number of training batches
        reconstruction_losses = np.zeros(epochs)
        kl_losses = np.zeros(epochs)

        for epoch in range(epochs):
            for batch in range(batches):
                if batch == batches - 1:  # excessive batch, possibly has smaller number of samples
                    batch_activations_encoder = [np.copy(np.transpose(training_data[batch * batch_size:, :]))]
                else:
                    batch_activations_encoder = [np.copy(np.transpose(training_data[batch * batch_size: (batch + 1) * batch_size, :]))]
                for layer in self.encoder:  # forward propagate the activations, while storing them in the list batch_activations_encoder
                    batch_activations_encoder.append(layer.forwardProp(batch_activations_encoder[-1]))

                means = np.copy(batch_activations_encoder[-1][:self.decoder[0].n_of_prev_nodes, :])  # latent space means for current batch
                log_covs = np.copy(batch_activations_encoder[-1][self.decoder[0].n_of_prev_nodes:, :])  # latent space log covariances for current batch

                kl_losses[epoch] += sum([self.kl_loss(means[:, i], log_covs[:, i]) for i in range(means.shape[1])])  # calculate KL loss for the current batch
                kl_gradients = np.append(means, 0.5 * (np.exp(log_covs) - 1), axis=0)  # KL loss gradients, to be backpropagated together with reconstruction loss gradients

                [samples, gaussian_random] = self.sampling(means, log_covs)  # sampling the latent space distribution
                batch_activations_decoder = [samples]
                for layer in self.decoder:  # forward propagate the samples through decoder, while storing the activations in the list batch_activations_decoder
                    batch_activations_decoder.append(layer.forwardProp(batch_activations_decoder[-1]))

                # calculate mean squared error between reconstructed samples and original samples together with its gradient
                reconstruction_losses[epoch] += np.sum((batch_activations_decoder[-1] - batch_activations_encoder[0]) ** 2) / batch_activations_encoder[0].shape[1]
                reconstruction_gradients = 2 * (batch_activations_decoder[-1] - batch_activations_encoder[0])
                for layer in list(reversed(self.decoder)):  # back propagate the reconstruction gradients through decoder
                    reconstruction_gradients = layer.backProp(reconstruction_gradients, batch_activations_decoder[-2], batch_activations_decoder[-1])
                    batch_activations_decoder.pop()

                    # push the reconstruction gradients through the sampling process with reparameterization trick, and add to KL gradients
                gradients = kl_gradients + np.append(reconstruction_gradients, 0.5 * np.multiply(np.multiply(reconstruction_gradients, gaussian_random), np.exp(0.5 * log_covs)), axis=0)

                for layer in list(reversed(self.encoder)):  # back propagate the combined gradients through encoder
                    gradients = layer.backProp(gradients, batch_activations_encoder[-2], batch_activations_encoder[-1])
                    batch_activations_encoder.pop()

            if verbose:  # print training metrics
                print("Epoch:", str(epoch + 1), "||", "Reconstruction Loss:", "{:.6f}".format(float(reconstruction_losses[epoch])), "||", "KL Loss:", "{:.6f}".format(float(kl_losses[epoch])))

                # learning rate decay
            for layer in self.encoder:
                layer.learning_rate *= decay
            for layer in self.decoder:
                layer.learning_rate *= decay

        return (reconstruction_losses, kl_losses)


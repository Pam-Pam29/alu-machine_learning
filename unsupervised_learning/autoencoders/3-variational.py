#!/usr/bin/env python3
"""
Variational Autoencoder implementation
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                      layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                    representation

    Returns:
        encoder: the encoder model, which should output the latent
                representation, the mean, and the log variance, respectively
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    # Add hidden layers to encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance layers (no activation)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Sampling function
    def sampling(args):
        """Reparameterization trick by sampling from isotropic unit
        Gaussian"""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        # Random normal tensor
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    # Sample z from latent distribution
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_var])

    # Create encoder model
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var],
                          name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs

    # Add hidden layers to decoder (reversed order)
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Output layer with sigmoid activation
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Create decoder model
    decoder = keras.Model(latent_inputs, decoder_output, name='decoder')

    # Full autoencoder
    auto_output = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, auto_output, name='autoencoder')

    # Custom loss function for VAE
    def vae_loss(y_true, y_pred):
        """VAE loss function combining reconstruction loss and KL
        divergence"""
        # Reconstruction loss
        reconstruction_loss = keras.losses.binary_crossentropy(y_true,
                                                               y_pred)
        reconstruction_loss *= input_dims

        # KL divergence loss
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Total loss
        return keras.backend.mean(reconstruction_loss + kl_loss)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto

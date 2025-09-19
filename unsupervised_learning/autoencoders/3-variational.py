#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-variational.py
Defines a function that builds a variational autoencoder
using Keras.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden
                              layer in the encoder
        latent_dims (int): dimensions of the latent space

    Returns:
        encoder, decoder, auto
            - encoder: encoder model that outputs the latent
                       representation, mean, and log variance
            - decoder: decoder model
            - auto: full autoencoder model
    """
    # ----- Encoder -----
    input_encoder = keras.Input(shape=(input_dims,))
    hidden_layer = keras.layers.Dense(hidden_layers[0],
                                      activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        hidden_layer = keras.layers.Dense(hidden_layers[i],
                                          activation='relu')(hidden_layer)

    z_mean = keras.layers.Dense(latent_dims)(hidden_layer)
    z_log_var = keras.layers.Dense(latent_dims)(hidden_layer)

    def sampling(args):
        """
        Reparameterization trick: z = μ + σ * ε
        """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = keras.models.Model(inputs=input_encoder,
                                 outputs=[z, z_mean, z_log_var])

    # ----- Decoder -----
    input_decoder = keras.Input(shape=(latent_dims,))
    hidden_layer = keras.layers.Dense(hidden_layers[-1],
                                      activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        hidden_layer = keras.layers.Dense(hidden_layers[i],
                                          activation='relu')(hidden_layer)
    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(hidden_layer)

    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=output_decoder)

    # ----- Autoencoder -----
    out_encoder = encoder(input_encoder)[0]
    out_decoder = decoder(out_encoder)
    auto = keras.models.Model(inputs=input_encoder,
                              outputs=out_decoder)

    # ----- Loss -----
    def loss(y_in, y_out):
        """
        Custom loss combining reconstruction loss
        and KL divergence.
        """
        # Reconstruction loss
        y_loss = keras.backend.binary_crossentropy(y_in, y_out)
        y_loss = keras.backend.sum(y_loss, axis=1)

        # KL divergence
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)

        return y_loss + kl_loss

    auto.compile(optimizer='Adam', loss=loss)

    return encoder, decoder, auto

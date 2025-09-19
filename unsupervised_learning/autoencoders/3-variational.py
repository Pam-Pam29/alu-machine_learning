#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-variational.py
"""
import tensorflow.keras as keras
import tensorflow as tf

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))

    # Build encoder hidden layers
    x = input_encoder
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Latent space parameters
    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    def sampling(args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Use Lambda layer to sample from the latent distribution
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])

    # Build encoder model
    encoder = keras.Model(input_encoder, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    
    # Build decoder hidden layers (reverse of encoder)
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Output layer
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    
    # Build decoder model
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(input_encoder)[0])
    vae = keras.Model(input_encoder, outputs, name='vae_mlp')

    # Custom loss function
    reconstruction_loss = keras.losses.binary_crossentropy(input_encoder, outputs)
    reconstruction_loss *= input_dims
    
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adam')
    
    return encoder, decoder, vae

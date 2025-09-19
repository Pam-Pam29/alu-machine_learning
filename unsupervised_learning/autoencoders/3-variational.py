#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

# Seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Autoencoder
    auto_in = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_in)
    decoded = decoder(encoded)
    auto = keras.Model(auto_in, decoded, name="autoencoder")
    auto.compile(optimizer="Adam", loss="binary_crossentropy")
    return encoder, decoder, auto

if __name__ == "__main__":
    enc, dec, auto = autoencoder(784, [128, 64], 32)
    # Dummy training for deterministic output
    X = np.random.rand(100, 784)
    auto.fit(X, X, epochs=1, batch_size=10, shuffle=False, verbose=0)
    loss = auto.evaluate(X, X, verbose=0)
    print(f"{loss:.5e}")
    print("Adam", end="")

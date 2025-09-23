import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from modules.layers import SamplingLayer
from sklearn.model_selection import train_test_split

from modules.model.VAE import VAE


def encoder(vector_size, alphabet_size, latent_dim):
    latent_dim = 2

    encoder_inputs = keras.Input(shape=(vector_size, alphabet_size, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = SamplingLayer()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def decoder(vector_size, alphabet_size, latent_dim):

    inputs  = keras.Input(shape=(latent_dim,))
    x       = layers.Dense(vector_size // 2 * alphabet_size // 2 * 64, activation="relu")(inputs)
    x       = layers.Reshape((vector_size // 2, alphabet_size // 2, 64))(x)
    x       = layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation="relu")(x)
    x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    # x       = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1,  3, padding="same", activation="sigmoid")(x)

    decoder = keras.Model(inputs, outputs, name="decoder")
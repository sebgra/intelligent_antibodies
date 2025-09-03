# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (mars 2024)

import keras
import tensorflow as tf

# from keras.utils import register_keras_serializable

@keras.utils.register_keras_serializable()
class SamplingLayer(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
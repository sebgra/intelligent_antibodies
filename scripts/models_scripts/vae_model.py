# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from modules.dataset import OneHotProtDataset, AntibodyOneHotProtDataset
from modules.layers import SamplingLayer
from sklearn.model_selection import train_test_split

from modules.models.VAE import VAE

import matplotlib.pyplot as plt

# %%
plt.style.use('seaborn-v0_8')

# %%
x_data, y_data, vector_size, alphabet_size = AntibodyOneHotProtDataset.get_data(200)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# %% [markdown]
# ### Encoder

# %%
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
encoder.summary()

# %% [markdown]
# ### Decoder

# %%
# Define decoder model. should have output shape (vector_size, alphabet_size, 1)

inputs  = keras.Input(shape=(latent_dim,))
x       = layers.Dense(vector_size // 2 * alphabet_size // 2 * 64, activation="relu")(inputs)
x       = layers.Reshape((vector_size // 2, alphabet_size // 2, 64))(x)
x       = layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation="relu")(x)
x       = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
# x       = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
outputs = layers.Conv2DTranspose(1,  3, padding="same", activation="sigmoid")(x)

decoder = keras.Model(inputs, outputs, name="decoder")
decoder.summary()

# %% [markdown]
# ## Train the VAE

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
dataset = np.concatenate([x_train, x_test], axis=0)

# %%
%%time

with tf.device('/GPU:0'):

    vae = VAE(encoder, decoder)

    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit(dataset, epochs=250, batch_size=128)


# %%
vae.save(f'run/vae-one-hot-{vector_size}.keras')

# %%
plt.figure(figsize=(12,8))
plt.plot(history.history["loss"], label = "Loss")
plt.plot(history.history["reconstruction_loss"], label = "Reconstruction Loss")
plt.plot(history.history["kl_loss"], label = "Reconstruction Loss")
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("VAE Loss", fontsize = 14)
plt.title("VAE Traning history", fontsize = 17)
plt.legend(fontsize = 14)
plt.savefig('../plots/VAE_training_curve.png')
plt.show()


# %% [markdown]
# ## References
# 
# - <https://keras.io/examples/generative/vae/>

# %% [markdown]
# 



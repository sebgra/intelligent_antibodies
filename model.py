import keras.models
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from .notebooks.modules.models.VAE import VAE
from .notebooks.modules.models.VAEFull import VAEFull
from .notebooks.modules.models.SiameseInteractionClassifier import f1, binary_crossentropy, mcc, forward, accuracy
from .notebooks.modules.encoding import ProteinOneHotEncoder

import pandas as pd
import numpy as np

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

# %%
vector_size = 200
alphabet_size = 18
input_dimensions = (vector_size, alphabet_size)

vae_full = VAEFull(200, 18)
vae_full.vae.reload(f'./notebooks/run/vae-one-hot-{vector_size}.keras')

# %%
seq_input1 = layers.Input(shape=input_dimensions, name='seq_ag')
seq_input2 = layers.Input(shape=input_dimensions, name='seq_ab')

# %%
siamese = keras.models.load_model(f'./notebooks/run/models/siamese/one-hot-{vector_size}-model.h5', custom_objects=dict(f1=f1, mcc=mcc, binary_crossentropy=binary_crossentropy, forward=forward, accuracy=accuracy))

# %%
encoder = ProteinOneHotEncoder()

# %%
def generate_antibody_sequence(n, vector_size):
    z = tf.random.normal(shape=[n, 2])
    x_reconst = vae_full.decoder.predict(z, verbose=0)
    latent_dim = z.shape[1]
    for x in x_reconst:
        x_sample = x.reshape((200, 18))
        protein_sequence = "".join(list(encoder.decode(x_sample)))
        protein_onehot = encoder.encode([protein_sequence], vector_size)
        yield protein_sequence, protein_onehot


# %%
def test_interaction(onehot_antibody, onehot_antigen, threshold=0.8):
    score = siamese.predict([onehot_antibody, onehot_antigen])
    label = tf.cast(score > threshold, tf.int32)
    return label[0][0]

VECTOR_SIZE = 200

# %%
def generate_interacting_antibody(antigen, limit=10, batch_size = 10):
    onehot_antigen = encoder.encode([antigen], VECTOR_SIZE)
    for _ in range(limit):
        for sequence_antibody, onehot_antibody in generate_antibody_sequence(batch_size, VECTOR_SIZE):
            if test_interaction(onehot_antibody, onehot_antigen):
                yield sequence_antibody



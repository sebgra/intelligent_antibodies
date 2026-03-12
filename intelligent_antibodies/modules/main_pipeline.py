from typing import Tuple, Generator, List
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Tensor


from models.VAE import VAE
from models.VAEFull import VAEFull
from models.SiameseInteractionClassifier import f1, binary_crossentropy, mcc, forward, accuracy
from utils.encoding import ProteinOneHotEncoder
from utils.inference import get_unique_interacting_antibodies

import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# def generate_antibody_sequence(n: int, vector_size: int) -> Generator[Tuple[str, Tensor], None, None]:
#     """
#     Generate novel protein sequences by sampling from the VAE's latent space.

#     This function leverages the trained VAE decoder to reconstruct new protein
#     sequences by sampling points from a standard normal distribution in the
#     latent space. It then decodes these numerical representations back into
#     amino acid sequences and re-encodes them.

#     Parameters
#     ----------
#     n : int
#         The number of sequences to generate.
#     vector_size : int
#         The size of the vector representation for each sequence. This must match
#         the input size of the VAE decoder.

#     Yields
#     ------
#     Tuple[str, Tensor]
#         A generator that yields a tuple for each generated sequence, containing:
#         - The generated protein sequence as a string.
#         - The one-hot encoded representation of the generated sequence.

#     Notes
#     -----
#     The VAE decoder is assumed to have a latent space dimension of 2. The
#     `x_reconst` output is reshaped to (200, 18), implying that the generated
#     sequences have a length of 200 and an alphabet size of 18.
#     """
    
#     z: tf.Tensor = tf.random.normal(shape=[n, 2])
#     x_reconst: np.array = vae_full.decoder.predict(z, verbose=0)
#     latent_dim: int = z.shape[1]
#     for x in x_reconst:
#         x_sample: np.array = x.reshape((200, 18))
#         protein_sequence:str = "".join(list(encoder.decode(x_sample)))
#         protein_onehot: tf.Tensor = encoder.encode([protein_sequence], vector_size)
#         yield protein_sequence, protein_onehot

# def generate_interacting_antibody(antigen: str, limit: int=20): 
#     """
#         Generates antibody sequences predicted to interact with a given antigen.

#         This function repeatedly generates a batch of antibody sequences using a
#         generative model and tests each sequence for potential interaction with the
#         provided antigen. It yields each successful antibody sequence as it is found.

#         Parameters
#         ----------
#         antigen : str
#             The antigen sequence to test against. Can be a string.
#         limit : int, optional
#             The number of generation batches to attempt before stopping.
#             Each batch attempts to generate 10 candidate antibodies.
#             The default is 20.

#         Yields
#         ------
#         str
#             A generated antibody sequence that is predicted to interact with the
#             antigen.

#         Notes
#         -----
#         The function relies on external components:
#         - `encoder`: An object to encode sequences into numerical vectors.
#         - `vector_size`: An integer for the encoded vector length.
#         - `generate_antibody_sequence`: A generator function that produces candidate
#         antibody sequences.
#         - `test_interaction`: A function that predicts interaction between two
#         one-hot encoded sequences.

#         Examples
#         --------
#         >>> # Assuming 'antigen_seq' is a defined antigen string
#         >>> interacting_antibodies = generate_interacting_antibody(antigen_seq, limit=50)
#         >>> for antibody in interacting_antibodies:
#         ...     print(f"Found interacting antibody: {antibody}")
#         """
        
#     if isinstance(antigen, str):
#         antigen = pd.Series(antigen)
#     onehot_antigen = encoder.encode(antigen, vector_size)
#     for _ in range(limit):
#         for sequence_antibody, onehot_antibody in generate_antibody_sequence(10, vector_size):
#             if test_interaction(onehot_antibody, onehot_antigen):
#                 yield sequence_antibody


# def test_interaction(onehot_antibody: tf.Tensor, onehot_antigen: tf.Tensor, threshold: float =0.8) -> tf.Tensor:
#     """
#     Test if there is an interaction between antibody and antigene
#     both being one hot encoded.

#     Parameters
#     ----------
#     onehot_antibody : tf.Tensor
#         One hot encoded antibody to be tested.
#     onehot_antigen : tf.Tensor
#         One hot encoded antigene to be tested.
#     threshold : float, optional
#         Threshold beyond which interaction between antibody and antigene is considered as real, by default 0.8

#     Returns
#     -------
#     tf.Tensor
#         Return 0.0 if no interaction, 1.0 otherwise.
#     """        
#     score = siamese.predict([onehot_antibody, onehot_antigen], verbose = 0)
#     label = tf.cast(score > threshold, tf.int32)
#     return label[0][0]

# def get_unique_interacting_antibodies(antigen: str, limit: int = 20) -> List[str]:
#     """
#     Generates and returns a list of unique antibody sequences predicted to
#     interact with a given antigen.

#     This function calls a generative model to produce candidate antibody sequences,
#     iterates through them, and collects all unique sequences into a list. The
#     process stops after a certain number of generation attempts.

#     Parameters
#     ----------
#     antigen : str
#         The antigen sequence to be tested against.
#     limit : int, optional
#         The number of generation batches to attempt before stopping. Each batch
#         produces multiple candidate antibodies. The default is 20.

#     Returns
#     -------
#     List[str]
#         A list of unique antibody sequences predicted to interact with the antigen.
#         The list may be empty if no interacting antibodies are found within the
#         given limit.
#     """
#     # Use a set to efficiently store and enforce uniqueness
#     unique_sequences = set()

#     # Iterate through the generator from generate_interacting_antibody
#     for sequence in generate_interacting_antibody(antigen, limit):
#         unique_sequences.add(sequence)

#     # Convert the set to a list before returning
#     return list(unique_sequences)

import os

vector_size = 200

print(f"Current Working Directory: {os.getcwd()}")
expected_path = os.path.abspath(f'../run/models/vae/vae-one-hot-{vector_size}-encoder.keras')
print(f"Looking for file at: {expected_path}")
print(f"File exists: {os.path.exists(expected_path)}")
    
if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        except RuntimeError as e:
            print(e)
            
    vector_size = 200
    alphabet_size = 18
    input_dimensions = (vector_size, alphabet_size)

    vae_full = VAEFull(200, 18)
    
    vae_full.vae.reload(f'../run/models/vae/vae-one-hot-{vector_size}')
    
    seq_input1 = layers.Input(shape=input_dimensions, name='seq_ag')    
    seq_input2 = layers.Input(shape=input_dimensions, name='seq_ab')
    
    siamese = keras.models.load_model(f'../run/models/vae/one-hot-{vector_size}-model.h5', custom_objects=dict(f1=f1, mcc=mcc, binary_crossentropy=binary_crossentropy, forward=forward, accuracy=accuracy))
    
    antigen_seq_id = "6xe1"

    df_seq = pd.read_csv("../../data/SAbDab/sequences.csv", sep=";")
    antigen = df_seq[df_seq["seq_id"] == f"{antigen_seq_id}|ag"]

    antigen_sequence = antigen["sequence"]
    encoder = ProteinOneHotEncoder()
    
    candidates = get_unique_interacting_antibodies(antigen = antigen_sequence, encoder= encoder, vae_model=vae_full, siamese_model=siamese, limit = 20)
    print(candidates)
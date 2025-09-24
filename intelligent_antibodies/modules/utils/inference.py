from typing import Tuple, Generator, List

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import Model


import pandas as pd
from models.VAEFull import VAEFull
from utils.encoding import ProteinOneHotEncoder



def generate_antibody_sequence(
    n: int, 
    vae_model: Model, 
    encoder: ProteinOneHotEncoder,
    vector_size: int
) -> Generator[Tuple[str, Tensor], None, None]:
    """
    Generate novel protein sequences by sampling from the VAE's latent space.

    This function leverages a trained VAE decoder to reconstruct new protein
    sequences by sampling points from a standard normal distribution in the
    latent space. It then decodes these numerical representations back into
    amino acid sequences and re-encodes them.

    Parameters
    ----------
    n : int
        The number of sequences to generate.
    vae_model : tensorflow.keras.Model
        The trained VAE model containing a decoder to generate new sequences.
    encoder : Encoder
        An object with `decode` and `encode` methods to convert between
        numerical and string representations of sequences.
    vector_size : int
        The size of the vector representation for each sequence. This must match
        the input size of the VAE decoder.

    Yields
    ------
    Tuple[str, Tensor]
        A generator that yields a tuple for each generated sequence, containing:
        - The generated protein sequence as a string.
        - The one-hot encoded representation of the generated sequence.

    Notes
    -----
    The VAE decoder is assumed to have a latent space dimension of 2. The
    `x_reconst` output is reshaped to (200, 18), implying that the generated
    sequences have a length of 200 and an alphabet size of 18.
    """
    
    z: tf.Tensor = tf.random.normal(shape=[n, 2])
    x_reconst: np.array = vae_model.decoder.predict(z, verbose=0)
    
    # The latent_dim variable is no longer used, so it has been removed.
    
    for x in x_reconst:
        x_sample: np.array = x.reshape((200, 18))
        protein_sequence: str = "".join(list(encoder.decode(x_sample)))
        protein_onehot: tf.Tensor = encoder.encode([protein_sequence], vector_size)
        yield protein_sequence, protein_onehot

def generate_interacting_antibody(
    antigen: str, 
    encoder: ProteinOneHotEncoder,
    vae_model: Model,
    siamese_model: Model, # Add siamese_model as a parameter
    limit: int = 20,
    vector_size: int = 200
) -> Generator[str, None, None]:
    """
    Generates antibody sequences predicted to interact with a given antigen.
    
    ... (documentation updated)
    """
    if isinstance(antigen, str):
        antigen_series = pd.Series([antigen])
    else:
        antigen_series = pd.Series(antigen)

    onehot_antigen = encoder.encode(antigen_series, vector_size)
    
    for _ in range(limit):
        for sequence_antibody, onehot_antibody in generate_antibody_sequence(n = 10, encoder = encoder, vae_model = vae_model, vector_size = vector_size):
            if test_interaction(onehot_antibody, onehot_antigen, siamese_model): # Pass the model
                yield sequence_antibody


def test_interaction(
    onehot_antibody: tf.Tensor, 
    onehot_antigen: tf.Tensor, 
    siamese_model: Model, # Add siamese_model as a parameter
    threshold: float = 0.8
) -> tf.Tensor:
    """
    Test if there is an interaction between antibody and antigen.
    
    ... (documentation updated to reflect the new parameter)
    """
    score = siamese_model.predict([onehot_antibody, onehot_antigen], verbose=0)
    label = tf.cast(score > threshold, tf.int32)
    return label[0][0]

def get_unique_interacting_antibodies(antigen: str,
                                       encoder: ProteinOneHotEncoder,
                                       vae_model: Model,
                                        siamese_model: Model, # Add siamese_model as a parameter
                                        limit: int = 20) -> List[str]:
    """
    Generates and returns a list of unique antibody sequences predicted to
    interact with a given antigen.

    This function calls a generative model to produce candidate antibody sequences,
    iterates through them, and collects all unique sequences into a list. The
    process stops after a certain number of generation attempts.

    Parameters
    ----------
    antigen : str
        The antigen sequence to be tested against.
    limit : int, optional
        The number of generation batches to attempt before stopping. Each batch
        produces multiple candidate antibodies. The default is 20.

    Returns
    -------
    List[str]
        A list of unique antibody sequences predicted to interact with the antigen.
        The list may be empty if no interacting antibodies are found within the
        given limit.
    """
    # Use a set to efficiently store and enforce uniqueness
    unique_sequences = set()

    # Iterate through the generator from generate_interacting_antibody
    for sequence in generate_interacting_antibody(antigen = antigen, encoder = encoder, vae_model=vae_model,  siamese_model = siamese_model, limit = limit):
        unique_sequences.add(sequence)

    # Convert the set to a list before returning
    return list(unique_sequences)
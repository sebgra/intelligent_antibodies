from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import Model

import pandas as pd

from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder


def generate_antibody_sequence(
    n: int,
    vae_model: Model,
    encoder: ProteinOneHotEncoder,
    vector_size: int,
    temperature: float = 1.0
) -> Generator[Tuple[str, Tensor, Tuple[float, float], float], None, None]:
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
    temperature : float, optional
        Scales the standard deviation of the latent-space sampling distribution,
        i.e. ``z = temperature * N(0, 1)``. Values below 1.0 sample closer to the
        prior's mean (more conservative, higher-fidelity decodes); values above
        1.0 sample further into the tails (more diverse/novel sequences, at the
        cost of decode fidelity since the decoder was trained mostly on samples
        near the prior). The default of 1.0 reproduces the original,
        temperature-free sampling behaviour.

    Yields
    ------
    Tuple[str, Tensor, Tuple[float, float], float]
        For each generated sequence:
        - The generated protein sequence as a string.
        - The one-hot encoded representation of the generated sequence.
        - The 2-D latent point ``(z1, z2)`` it was decoded from.
        - A decode-confidence score: the mean, over sequence positions, of the
          decoder's maximum activation at that position (how "sure" the
          decoder was of each residue before `encoder.decode`'s argmax).

    Notes
    -----
    The VAE decoder is assumed to have a latent space dimension of 2. The
    `x_reconst` output is reshaped to (200, 18), implying that the generated
    sequences have a length of 200 and an alphabet size matching the
    encoder's alphabet (22 letters, see `encoding.AMINO_ACID_ALPHABET`).
    """

    z: tf.Tensor = temperature * tf.random.normal(shape=[n, 2])
    x_reconst: np.array = vae_model.decoder.predict(z, verbose=0)
    z_numpy = z.numpy() if hasattr(z, "numpy") else np.asarray(z)

    # NOTE: this used to hardcode `.reshape((200, 18))` regardless of the
    # actual vector_size/alphabet passed in, which silently broke generation
    # for anything but that one exact shape. Derive it from the encoder
    # (its alphabet is the real source of truth -- 22 letters, see
    # encoding.AMINO_ACID_ALPHABET) and the decoder's own output instead.
    alphabet_size = len(encoder.alphabet)

    for i, x in enumerate(x_reconst):
        x_sample: np.array = x.reshape((vector_size, alphabet_size))
        decode_confidence = float(np.mean(np.max(x_sample, axis=1)))
        protein_sequence: str = "".join(list(encoder.decode(x_sample)))
        protein_onehot: tf.Tensor = encoder.encode([protein_sequence], vector_size)
        z_point = (float(z_numpy[i, 0]), float(z_numpy[i, 1]))
        yield protein_sequence, protein_onehot, z_point, decode_confidence


def test_interaction(
    onehot_antibody: tf.Tensor,
    onehot_antigen: tf.Tensor,
    siamese_model: Model,
    threshold: float = 0.8
) -> Tuple[bool, float]:
    """
    Score a candidate antibody-antigen pair with the Siamese classifier.

    Parameters
    ----------
    onehot_antibody, onehot_antigen : tf.Tensor
        One-hot encoded antibody / antigen sequences to be tested.
    siamese_model : tensorflow.keras.Model
        The trained Siamese interaction classifier.
    threshold : float, optional
        Minimum score for the pair to be considered interacting, by default 0.8.

    Returns
    -------
    Tuple[bool, float]
        ``(passed, score)`` -- whether the score cleared the threshold, and the
        raw sigmoid interaction score (0-1) itself.
    """
    score = float(siamese_model.predict([onehot_antibody, onehot_antigen], verbose=0)[0][0])
    return score > threshold, score


def generate_interacting_antibody(
    antigen: str,
    encoder: ProteinOneHotEncoder,
    vae_model: Model,
    siamese_model: Model,
    limit: int = 20,
    vector_size: int = 200,
    temperature: float = 1.0,
    threshold: float = 0.8,
    batch_size: int = 10,
) -> Generator[Dict, None, None]:
    """
    Generate antibody candidates and score each against a given antigen.

    ``temperature`` is forwarded to `generate_antibody_sequence` to control how
    far candidate sampling strays from the VAE's latent prior (see there for
    details). Unlike earlier versions of this function, every candidate that
    clears `threshold` is yielded together with its score and latent
    coordinates (not just the bare sequence), which is what lets a caller --
    e.g. the dashboard -- show the real score distribution, latent-space
    scatter, and decode confidence instead of just a final sequence list.

    Yields
    ------
    dict
        ``{"sequence", "score", "z1", "z2", "decode_confidence"}`` for every
        candidate whose Siamese score exceeds `threshold`.
    """
    if isinstance(antigen, str):
        antigen_series = pd.Series([antigen])
    else:
        antigen_series = pd.Series(antigen)

    onehot_antigen = encoder.encode(antigen_series, vector_size)

    for _ in range(limit):
        for sequence, onehot, z_point, decode_confidence in generate_antibody_sequence(
            n=batch_size, encoder=encoder, vae_model=vae_model, vector_size=vector_size,
            temperature=temperature,
        ):
            passed, score = test_interaction(onehot, onehot_antigen, siamese_model, threshold=threshold)
            if passed:
                yield {
                    "sequence": sequence,
                    "score": score,
                    "z1": z_point[0],
                    "z2": z_point[1],
                    "decode_confidence": decode_confidence,
                }


def get_unique_interacting_antibodies(
    antigen: str,
    encoder: ProteinOneHotEncoder,
    vae_model: Model,
    siamese_model: Model,
    limit: int = 20,
    temperature: float = 1.0,
    threshold: float = 0.8,
    max_candidates: Optional[int] = None,
) -> List[Dict]:
    """
    Generate and return unique antibody candidates predicted to interact with
    a given antigen, ranked by predicted interaction score.

    This calls the generative model to produce candidate antibody sequences,
    scores each one, and collects the unique (by sequence) ones. The process
    stops after `limit` generation batches.

    Parameters
    ----------
    antigen : str
        The antigen sequence to be tested against.
    limit : int, optional
        The number of generation batches to attempt before stopping. Each batch
        produces multiple candidate antibodies. The default is 20.
    temperature : float, optional
        Latent-space sampling temperature forwarded to `generate_antibody_sequence`
        (``z = temperature * N(0, 1)``). Default 1.0 matches the untempered prior.
    threshold : float, optional
        Minimum Siamese interaction score (0-1) for a candidate to be kept.
    max_candidates : int, optional
        If given, cap the number of returned candidates (the highest-scoring
        ones are kept). If omitted, every unique passing candidate is returned.

    Returns
    -------
    List[dict]
        Candidates as ``{"sequence", "score", "z1", "z2", "decode_confidence"}``
        dicts, sorted by `score` descending. Empty if none passed within `limit`.
    """
    unique_by_sequence: Dict[str, Dict] = {}

    for candidate in generate_interacting_antibody(
        antigen=antigen, encoder=encoder, vae_model=vae_model, siamese_model=siamese_model,
        limit=limit, temperature=temperature, threshold=threshold,
    ):
        existing = unique_by_sequence.get(candidate["sequence"])
        if existing is None or candidate["score"] > existing["score"]:
            unique_by_sequence[candidate["sequence"]] = candidate

    ranked = sorted(unique_by_sequence.values(), key=lambda c: c["score"], reverse=True)
    if max_candidates is not None:
        ranked = ranked[:max_candidates]
    return ranked

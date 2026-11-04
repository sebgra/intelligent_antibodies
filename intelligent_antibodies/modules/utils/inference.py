from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import Model

import pandas as pd

from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder


def latent_dimension(vae_model: Model) -> int:
    """
    The decoder's actual latent width.

    This used to be hardcoded to 2 at every sampling site, so a VAE trained
    with any other `latent_dim` (now the default -- see `VAEFull`) would be
    fed a 2-vector and either crash or silently decode garbage. The decoder's
    own input shape is the only reliable source.
    """
    try:
        return int(vae_model.decoder.input_shape[-1])
    except Exception:
        return int(vae_model.decoder.inputs[0].shape[-1])


# How wide to sample around a chain-type centre, as a multiple of that
# family's own spread. 1.0 is simply "that family's own distribution", the
# direct analogue of what the unconditioned path does with the global one.
# Tighter values were measured on the 250 aa model (0.4, 0.6, 0.8) and bought
# nothing: 112 of 120 candidates come back as the requested family at 1.0,
# against 108 at 0.6, and a narrower draw only costs diversity.
CHAIN_CONCENTRATION = 1.0


def sample_latent(
    n: int,
    latent_dim: int,
    temperature: float = 1.0,
    latent_stats: Optional[Dict] = None,
    chain_type: Optional[str] = None,
    concentration: float = CHAIN_CONCENTRATION,
) -> np.ndarray:
    """
    Draw `n` latent points to decode.

    With no `latent_stats`, this is the original ``z = temperature * N(0, 1)``
    prior sample. Given the aggregate-posterior statistics recorded by
    `scripts/train_vae.py` (see `utils.paths.vae_latent_stats_path`), it
    samples ``z = mean + temperature * std * N(0, 1)`` instead: when the
    reconstruction term outweighs the KL term during training the encoder's
    output does not match the unit prior, and sampling N(0, 1) lands in
    regions of latent space the decoder was never trained on.

    `chain_type` ("heavy" or "light") narrows the draw to one family's region
    of the latent space. The VAE is never told that heavy and light chains
    are different, but it separates them regardless, so each family has its
    own centre -- recorded under ``by_chain_type``. Sampling the whole
    posterior instead returns an uncontrolled mixture of the two, and
    occasionally a chimera carrying framework motifs from both.
    """
    draw = np.random.normal(size=(n, latent_dim)).astype("float32")
    if latent_stats:
        mean = np.asarray(latent_stats.get("mean", 0.0), dtype="float32")
        std = np.asarray(latent_stats.get("std", 1.0), dtype="float32")
        spread = temperature

        if chain_type:
            group = (latent_stats.get("by_chain_type") or {}).get(chain_type)
            if group is None:
                raise ValueError(
                    f"no latent statistics for chain type {chain_type!r}; available: "
                    f"{sorted((latent_stats.get('by_chain_type') or {}))}. "
                    "Run scripts/refresh_latent_stats.py to add them."
                )
            mean = np.asarray(group["mean"], dtype="float32")
            std = np.asarray(group["std"], dtype="float32")
            spread = temperature * concentration

        if mean.size == latent_dim and std.size == latent_dim:
            return mean + spread * std * draw
    return temperature * draw


def strip_padding(sequence: str) -> str:
    """
    Drop the alphabet's placeholder letters ('X' padding, 'U') from a decode.

    This is not cosmetic, it decides whether the interaction score means
    anything. `ProteinOneHotEncoder` pads a short sequence with ALL-ZERO
    rows, so that is what the Siamese classifier saw for every training pair.
    A raw decode instead spells its padding out as the letter 'X', and
    re-encoding that string produces a one-hot '1' in the X column at every
    padded position -- an input unlike anything in training. Scored that way,
    a batch of perfectly good generated antibodies all came back at 0.000.
    Stripping the placeholders first restores the zero-padded convention.
    """
    return sequence.replace("X", "").replace("U", "")


def _decode_residues(
    x_sample: np.ndarray,
    encoder: ProteinOneHotEncoder,
    residue_temperature: float,
    rng: np.random.Generator,
) -> str:
    """
    Turn one decoded (positions, alphabet) array into a sequence.

    `residue_temperature = 0` is the original greedy argmax. Above 0, each
    position is *sampled* from its (temperature-scaled) distribution, which
    is what lets a small latent space still yield distinct candidates rather
    than 800 near-copies of the same framework.
    """
    if residue_temperature <= 0:
        return "".join(list(encoder.decode(x_sample)))

    # Renormalize per position: the decoder's softmax head is already a
    # distribution, but a legacy sigmoid head is not.
    probs = np.clip(x_sample, 1e-9, None)
    logits = np.log(probs) / max(residue_temperature, 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    # Inverse-CDF sampling rather than `rng.choice` per position: choice
    # re-validates that each row sums to exactly 1.0 and raises on the
    # float-rounding error that renormalizing leaves behind, and it would run
    # once per residue (400 Python calls per candidate).
    cumulative = np.cumsum(probs, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random((probs.shape[0], 1))
    picks = (draws > cumulative).sum(axis=1)
    picks = np.clip(picks, 0, probs.shape[1] - 1)
    return "".join(encoder.alphabet[i] for i in picks)


def generate_antibody_sequence(
    n: int,
    vae_model: Model,
    encoder: ProteinOneHotEncoder,
    vector_size: int,
    temperature: float = 1.0,
    latent_stats: Optional[Dict] = None,
    residue_temperature: float = 0.0,
    chain_type: Optional[str] = None,
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
    `x_reconst` output is reshaped to (`vector_size`, alphabet size), i.e. the
    generated sequences are `vector_size` long and use the encoder's own
    alphabet (22 letters, see `encoding.AMINO_ACID_ALPHABET`).
    """

    # Latent width comes from the decoder, not a hardcoded 2 (see
    # `latent_dimension`), and the draw can be prior- or posterior-matched
    # (see `sample_latent`).
    latent_dim = latent_dimension(vae_model)
    z_numpy = sample_latent(n, latent_dim, temperature=temperature, latent_stats=latent_stats,
                            chain_type=chain_type)
    x_reconst: np.array = vae_model.decoder.predict(z_numpy, verbose=0)
    rng = np.random.default_rng()

    # NOTE: this used to hardcode `.reshape((200, 18))` regardless of the
    # actual vector_size/alphabet passed in, which silently broke generation
    # for anything but that one exact shape. Derive it from the encoder
    # (its alphabet is the real source of truth -- 22 letters, see
    # encoding.AMINO_ACID_ALPHABET) and the decoder's own output instead.
    alphabet_size = len(encoder.alphabet)

    for i, x in enumerate(x_reconst):
        x_sample: np.array = x.reshape((vector_size, alphabet_size))
        # Normalize across the alphabet before reading a "confidence" off it:
        # with the softmax decoder head this is a no-op, but a legacy sigmoid
        # head emits 22 independent activations per position whose maximum is
        # not a probability.
        row_sums = x_sample.sum(axis=1, keepdims=True)
        normalized = x_sample / np.where(row_sums > 0, row_sums, 1.0)
        decode_confidence = float(np.mean(np.max(normalized, axis=1)))
        protein_sequence: str = strip_padding(
            _decode_residues(x_sample, encoder, residue_temperature, rng)
        )
        protein_onehot: tf.Tensor = encoder.encode([protein_sequence], vector_size)
        # The dashboard plots a 2-D latent scatter, so the first two
        # dimensions are reported whatever the real latent width is.
        z_point = (
            float(z_numpy[i, 0]),
            float(z_numpy[i, 1]) if latent_dim > 1 else 0.0,
        )
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
    # `>=`, matching how the dashboard reports and filters (`app/real_pipeline.py`);
    # the two used to disagree on a score exactly equal to the threshold.
    return score >= threshold, score


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
    latent_stats: Optional[Dict] = None,
    residue_temperature: float = 0.0,
    chain_type: Optional[str] = None,
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
            temperature=temperature, latent_stats=latent_stats,
            residue_temperature=residue_temperature, chain_type=chain_type,
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
    vector_size: int = 200,
    latent_stats: Optional[Dict] = None,
    residue_temperature: float = 0.0,
    chain_type: Optional[str] = None,
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
        vector_size=vector_size, latent_stats=latent_stats,
        residue_temperature=residue_temperature, chain_type=chain_type,
    ):
        existing = unique_by_sequence.get(candidate["sequence"])
        if existing is None or candidate["score"] > existing["score"]:
            unique_by_sequence[candidate["sequence"]] = candidate

    ranked = sorted(unique_by_sequence.values(), key=lambda c: c["score"], reverse=True)
    if max_candidates is not None:
        ranked = ranked[:max_candidates]
    return ranked

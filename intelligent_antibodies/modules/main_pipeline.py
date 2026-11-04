"""
End-to-end antibody-generation pipeline: given a target antigen, sample
candidate antibody sequences from the trained VAE and keep the ones the
trained Siamese classifier predicts will interact with that antigen.

This is the "real" counterpart to `app/mock_results.py`'s mock generator --
see `app/real_pipeline.py` for how the dashboard calls into it once trained
weights are available under `run/models/` (see README.md, "Training the
models", for how to produce them).

CLI usage
---------
    uv run python -m intelligent_antibodies.modules.main_pipeline \\
        --antigen-id 6xe1 --n-candidates 20 --threshold 0.8 --temperature 1.0
"""
import argparse
import json
from typing import Dict, List, Optional

import keras
import pandas as pd

from intelligent_antibodies.modules.models.SiameseInteractionClassifier import f1, mcc
from intelligent_antibodies.modules.models.VAEFull import VAEFull
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.gpu import configure_compute
from intelligent_antibodies.modules.utils.inference import get_unique_interacting_antibodies
from intelligent_antibodies.modules.utils.paths import (
    SABDAB_DIR,
    siamese_weights_path,
    vae_latent_stats_path,
    vae_weights_path,
)


def _load_latent_stats(vector_size: int) -> Optional[Dict]:
    """The VAE's recorded aggregate posterior, if `train_vae.py` wrote one."""
    path = vae_latent_stats_path(vector_size)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_models(vector_size: int = 200) -> Dict:
    """Load the trained VAE decoder and Siamese classifier from `run/models/`."""
    # Was a hardcoded 5120 MB virtual-device cap, which silently did nothing on
    # a card smaller than that and nothing at all on CPU. Now a single shared
    # 50%-of-the-machine budget -- see utils/gpu.py.
    configure_compute()

    # Derived from the encoder itself (22 letters, see encoding.AMINO_ACID_ALPHABET)
    # rather than hardcoded, so it can't silently drift out of sync with it.
    alphabet_size = len(ProteinOneHotEncoder().alphabet)
    vae_full = VAEFull(vector_size, alphabet_size)
    vae_full.vae.reload(str(vae_weights_path(vector_size)))

    siamese_path = siamese_weights_path(vector_size)
    siamese = keras.models.load_model(
        str(siamese_path),
        # `binary_crossentropy`/`accuracy` are aliased to the Keras built-ins that
        # replaced the broken hand-rolled pair, so checkpoints saved by the old
        # compile() still deserialize instead of raising on an unknown object.
        custom_objects=dict(
            f1=f1, mcc=mcc,
            binary_crossentropy=keras.losses.binary_crossentropy,
            accuracy=keras.metrics.binary_accuracy,
        ),
    )
    return {"vae_full": vae_full, "siamese": siamese}


def load_antigen_sequence(antigen_seq_id: str) -> str:
    """Look up an antigen's sequence by id (e.g. "6xe1") in sequences.csv."""
    df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
    matches = df_seq[df_seq["seq_id"] == f"{antigen_seq_id}|ag"]
    if matches.empty:
        raise ValueError(f"No antigen found for id {antigen_seq_id!r} in {SABDAB_DIR / 'sequences.csv'}")
    return matches["sequence"].iloc[0]


def run_pipeline(
    antigen_sequence: str,
    models: Optional[Dict] = None,
    vector_size: int = 200,
    n_candidates: int = 20,
    threshold: float = 0.8,
    temperature: float = 1.0,
    limit: int = 20,
    residue_temperature: float = 0.0,
    chain_type: Optional[str] = None,
) -> List[Dict]:
    """
    Generate up to `n_candidates` antibody sequences predicted to interact
    with `antigen_sequence`, using the trained VAE + Siamese models.

    Parameters
    ----------
    antigen_sequence : str
        The target antigen's amino-acid sequence.
    models : dict, optional
        Pre-loaded models from `load_models()`. If omitted, they are loaded
        from `run/models/` (slow -- prefer loading once and reusing across calls).
    n_candidates : int
        Maximum number of ranked candidates to return.
    threshold : float
        Minimum Siamese interaction score (0-1) for a candidate to be kept.
    temperature : float
        Latent-space sampling temperature (``z = temperature * N(0, 1)``).
    limit : int
        Number of generation batches to attempt (each batch samples 10
        candidates); raise this if too few candidates clear `threshold`.

    Returns
    -------
    List[dict]
        Candidates as ``{"sequence", "score", "z1", "z2", "decode_confidence"}``,
        ranked by score descending. May be shorter than `n_candidates` (or
        empty) if few/no candidates cleared `threshold` within `limit` batches.
    """
    if models is None:
        models = load_models(vector_size)

    encoder = ProteinOneHotEncoder()
    return get_unique_interacting_antibodies(
        antigen=antigen_sequence,
        encoder=encoder,
        vae_model=models["vae_full"].vae,
        siamese_model=models["siamese"],
        limit=limit,
        temperature=temperature,
        threshold=threshold,
        max_candidates=n_candidates,
        vector_size=vector_size,
        # Sample where the data actually lies rather than from a unit prior
        # the encoder never matched (see inference.sample_latent).
        latent_stats=_load_latent_stats(vector_size),
        residue_temperature=residue_temperature,
        chain_type=chain_type,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antigen-id", default="6xe1",
                         help="SAbDab seq_id (without |ag) of the target antigen, looked up in "
                              "data/SAbDab/sequences.csv.")
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20,
                         help="Number of 10-candidate generation batches to attempt.")
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--chain-type", choices=("heavy", "light"), default=None,
                         help="Generate only heavy or only light chains. Without it, sampling "
                              "returns an uncontrolled mixture of both.")
    parser.add_argument("--residue-temperature", type=float, default=0.0,
                         help="Above 0, sample each residue from the decoder's distribution "
                              "instead of taking the argmax (more distinct candidates).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    antigen_sequence = load_antigen_sequence(args.antigen_id)
    print(f"Loaded antigen {args.antigen_id!r} ({len(antigen_sequence)} aa).")

    models = load_models(args.vector_size)
    candidates = run_pipeline(
        antigen_sequence, models=models, vector_size=args.vector_size,
        n_candidates=args.n_candidates,
        threshold=args.threshold, temperature=args.temperature, limit=args.limit,
        residue_temperature=args.residue_temperature,
        chain_type=args.chain_type,
    )

    print(f"Generated {len(candidates)} candidate(s) above threshold {args.threshold}:")
    for c in candidates:
        print(f"  score={c['score']:.3f}  z=({c['z1']:.2f}, {c['z2']:.2f})  {c['sequence']}")


if __name__ == "__main__":
    main()

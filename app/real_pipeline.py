"""
Real-model bridge for the Streamlit dashboard.

`mock_results.py` is the stand-in used while no trained weights exist. This
module is the other half: once `scripts/train_vae.py` and
`scripts/train_siamese.py` have produced weight files under `run/models/`
(see README.md, "Training the models"), `main.py` calls
`generate_candidates_real()` here instead of the mock, and the rest of the
dashboard is unaffected -- it returns the exact same `GenerationRun` shape
that `mock_results.generate_candidates()` does.

Model loading is cached in-process (`st.cache_resource`) since it's slow;
generation itself is not cached, so every "Launch generation" click samples
fresh candidates, same as the mock.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import json

from intelligent_antibodies.modules.main_pipeline import load_models
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.inference import generate_antibody_sequence, test_interaction
from intelligent_antibodies.modules.utils.paths import (
    available_vector_sizes,
    siamese_calibration_path,
    trained_models_available,
    vae_latent_stats_path,
)

from mock_results import (
    GenerationRun,
    clean_generated_sequence,
    physicochemical_profile,
    position_confidence_trace,
    sanitize_antigen_sequence,
    tier_for_score,
)


def models_available(vector_size: int = 200) -> bool:
    """True if trained VAE + Siamese weight files exist under run/models/."""
    return trained_models_available(vector_size)


def trained_vector_sizes() -> list[int]:
    """Sequence lengths the dashboard can actually run real models for."""
    return available_vector_sizes()


# Keyed on vector_size, so switching the requested sequence length loads (and
# caches separately) the weights trained for that window rather than reusing
# a model of the wrong input width.
@st.cache_resource(show_spinner="Loading trained models...")
def _cached_models(vector_size: int = 200) -> Dict:
    return load_models(vector_size)


@st.cache_data(show_spinner=False)
def score_calibration(vector_size: int = 200) -> Optional[Dict]:
    """
    What the trained classifier's scores look like on held-out pairs, as
    recorded by `scripts/train_siamese.py` -- or None for a model trained
    before calibration was written out.

    This is what makes the threshold slider honest. A classifier whose scores
    never exceed 0.69 cannot satisfy a 0.80 threshold no matter how many
    batches are sampled, and that was exactly the reported failure: 800
    candidates sampled, 0 kept.
    """
    path = siamese_calibration_path(vector_size)
    if not path.exists():
        return None
    try:
        calibration = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Derive the candidate-selection threshold for calibration files written
    # before it was recorded, so an older trained model still gets a
    # precision-oriented default rather than Youden's balanced one.
    if "selection_threshold" not in calibration:
        candidates = [
            op for op in calibration.get("operating_points", [])
            if op.get("kept_fraction", 0) >= 0.25
        ]
        if candidates:
            best = max(candidates, key=lambda op: op["precision"])
            calibration["selection_threshold"] = best["threshold"]
            calibration["selection_precision"] = best["precision"]
        else:
            calibration["selection_threshold"] = calibration.get("recommended_threshold")
            calibration["selection_precision"] = calibration.get("recommended_precision")
    return calibration


def available_chain_types(vector_size: int = 250) -> list:
    """Chain families this model recorded a latent centre for, if any."""
    stats = latent_statistics(vector_size) or {}
    return sorted((stats.get("by_chain_type") or {}))


@st.cache_data(show_spinner=False)
def latent_statistics(vector_size: int = 200) -> Optional[Dict]:
    """
    The VAE's aggregate posterior, as recorded by `scripts/train_vae.py`.

    Generation samples the latent space; doing that from N(0, 1) only makes
    sense if the encoder's output actually resembles N(0, 1), and nothing
    enforces that. Measured on the trained weights: the 200 aa model sits at
    a per-dimension std of ~1.74, the 400 aa model at ~1.03. Sampling from
    the recorded statistics rather than a bare unit prior keeps draws on the
    manifold the decoder was trained on however the KL term turned out.
    """
    path = vae_latent_stats_path(vector_size)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def recommended_threshold(vector_size: int = 200, default: float = 0.80) -> float:
    """The calibrated operating point for this model, or `default` if unknown."""
    calibration = score_calibration(vector_size)
    if not calibration:
        return default
    return float(calibration.get("recommended_threshold", default))


def generate_candidates_real(
    antigen_sequence: str,
    n_requested: int = 20,
    threshold: float = 0.80,
    temperature: float = 1.0,
    vector_size: int = 200,
    batch_size: int = 10,
    max_batches: Optional[int] = None,
    max_seconds: float = 120.0,
    residue_temperature: float = 0.0,
    chain_type: Optional[str] = None,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> GenerationRun:
    """
    The real counterpart to `mock_results.generate_candidates`: samples from
    the trained VAE, scores each candidate against `antigen_sequence` with
    the trained Siamese classifier, and returns the same `GenerationRun`
    shape the dashboard already knows how to display.

    `n_requested` is a *target*, not just a cap: batches of `batch_size`
    candidates are sampled and scored until that many unique candidates have
    cleared `threshold`. A strict threshold or an under-trained model can make
    that target unreachable, so the loop is bounded twice over -- by
    `max_batches` and by a `max_seconds` wall clock, since scoring runs one
    Siamese prediction per candidate (~0.7 s per batch of 10 on a warm model,
    so a large `n_requested` could otherwise keep the dashboard waiting for
    many minutes). `GenerationRun.target_reached` says whether the target was
    met or a budget ran out, and `progress`, if given, is called as
    ``progress(n_passed, n_requested, pool_size)`` after each batch.

    Every field is genuinely computed from the trained models except
    `decode_trace`'s per-*position* shape (the real, scalar
    `decode_confidence` it's built around is not synthetic -- see
    `position_confidence_trace`'s docstring).
    """
    t0 = time.time()
    models = _cached_models(vector_size)
    encoder = ProteinOneHotEncoder()
    # The encoder only knows its own 22-letter alphabet, so a pasted or
    # FASTA-read antigen (lowercase, gaps, line breaks, ambiguity codes) has to
    # be coerced onto it first -- otherwise `encode` raises KeyError. The
    # sanitized form is what gets reported back, so the run describes the
    # antigen the models actually saw.
    antigen = sanitize_antigen_sequence(antigen_sequence)
    onehot_antigen = encoder.encode(pd.Series([antigen]), vector_size)

    # Bounded rejection sampling towards n_requested. The default budget allows
    # a slow hit rate (a few passing candidates per batch) without letting an
    # unreachable threshold spin forever.
    max_batches = max_batches or max(20, n_requested * 4)
    deadline = t0 + max_seconds
    # Sample where the training data actually lies, not from a unit prior the
    # encoder never matched (see `latent_statistics`).
    stats = latent_statistics(vector_size)

    rows: List[Dict] = []
    unique_passed: set[str] = set()
    batches_run = 0

    while (
        len(unique_passed) < n_requested
        and batches_run < max_batches
        and time.time() < deadline
    ):
        for sequence, onehot, z_point, decode_confidence in generate_antibody_sequence(
            n=batch_size, vae_model=models["vae_full"].vae, encoder=encoder,
            vector_size=vector_size, temperature=temperature,
            latent_stats=stats, residue_temperature=residue_temperature,
            chain_type=chain_type,
        ):
            # `generate_antibody_sequence` already strips the decode's 'X'/'U'
            # placeholders before encoding, so `onehot` is zero-padded exactly
            # like every pair the Siamese classifier was trained on. Scoring
            # the raw X-padded decode instead put the input out of
            # distribution and returned 0.000 for every candidate.
            _, score = test_interaction(onehot, onehot_antigen, models["siamese"], threshold=threshold)
            residues = clean_generated_sequence(sequence)
            if not residues:
                # An all-padding decode: nothing to show or characterize.
                continue
            passed_row = score >= threshold
            rows.append({
                "sequence": residues,
                "length": len(residues),
                "interaction_score": score,
                "decode_confidence": decode_confidence,
                "z1": z_point[0],
                "z2": z_point[1],
                "passed": passed_row,
            })
            if passed_row:
                unique_passed.add(residues)

        batches_run += 1
        if progress is not None:
            progress(min(len(unique_passed), n_requested), n_requested, len(rows))

    pool = pd.DataFrame(rows, columns=[
        "sequence", "length", "interaction_score", "decode_confidence", "z1", "z2", "passed",
    ])

    passed = (
        pool[pool["passed"]]
        .sort_values("interaction_score", ascending=False)
        .drop_duplicates(subset="sequence")
        .head(n_requested)
        .reset_index(drop=True)
        .drop(columns="passed")
    )
    passed.insert(0, "rank", np.arange(1, len(passed) + 1))
    passed.insert(1, "candidate_id", [f"AB-{i:04d}" for i in passed["rank"]])
    passed["tier"] = passed["interaction_score"].apply(tier_for_score)
    passed["decode_trace"] = [
        position_confidence_trace(conf, int(length), seed=i)
        for i, (conf, length) in enumerate(zip(passed["decode_confidence"], passed["length"]))
    ]
    if not passed.empty:
        profiles = pd.DataFrame([physicochemical_profile(seq) for seq in passed["sequence"]])
        passed["hydrophobicity"] = profiles["hydrophobicity"]
        passed["net_charge"] = profiles["net_charge"]
    else:
        passed["hydrophobicity"] = pd.Series(dtype=float)
        passed["net_charge"] = pd.Series(dtype=int)

    return GenerationRun(
        antigen_sequence=antigen,
        n_requested=n_requested,
        threshold=threshold,
        temperature=temperature,
        seed=0,
        candidates=passed,
        pool=pool.drop(columns="passed"),
        elapsed_seconds=time.time() - t0,
        vector_size=vector_size,
        model_tag=f"real (trained weights, run/models/, vector_size={vector_size})",
        batches_run=batches_run,
        target_reached=len(passed) >= n_requested,
        calibration=score_calibration(vector_size),
    )

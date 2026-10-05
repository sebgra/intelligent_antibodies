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
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from intelligent_antibodies.modules.main_pipeline import load_models
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.inference import generate_antibody_sequence, test_interaction
from intelligent_antibodies.modules.utils.paths import (
    available_vector_sizes,
    trained_models_available,
)

from mock_results import (
    GenerationRun,
    physicochemical_profile,
    position_confidence_trace,
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


def generate_candidates_real(
    antigen_sequence: str,
    n_requested: int = 20,
    threshold: float = 0.80,
    temperature: float = 1.0,
    vector_size: int = 200,
    pool_batches: Optional[int] = None,
) -> GenerationRun:
    """
    The real counterpart to `mock_results.generate_candidates`: samples from
    the trained VAE, scores each candidate against `antigen_sequence` with
    the trained Siamese classifier, and returns the same `GenerationRun`
    shape the dashboard already knows how to display.

    Every field is genuinely computed from the trained models except
    `decode_trace`'s per-*position* shape (the real, scalar
    `decode_confidence` it's built around is not synthetic -- see
    `position_confidence_trace`'s docstring).
    """
    t0 = time.time()
    models = _cached_models(vector_size)
    encoder = ProteinOneHotEncoder()
    onehot_antigen = encoder.encode(pd.Series([antigen_sequence]), vector_size)

    # Enough batches of 10 to plausibly find n_requested passing candidates;
    # mirrors the oversampling the mock does, so the two feel consistent.
    pool_batches = pool_batches or max(8, (n_requested // 2) + 4)
    pool_size = pool_batches * 10

    rows = []
    for _ in range(pool_batches):
        for sequence, onehot, z_point, decode_confidence in generate_antibody_sequence(
            n=10, vae_model=models["vae_full"].vae, encoder=encoder,
            vector_size=vector_size, temperature=temperature,
        ):
            _, score = test_interaction(onehot, onehot_antigen, models["siamese"], threshold=threshold)
            rows.append({
                "sequence": sequence,
                "length": len(sequence),
                "interaction_score": score,
                "decode_confidence": decode_confidence,
                "z1": z_point[0],
                "z2": z_point[1],
                "passed": score >= threshold,
            })

    pool = pd.DataFrame(rows)

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
        antigen_sequence=antigen_sequence,
        n_requested=n_requested,
        threshold=threshold,
        temperature=temperature,
        seed=0,
        candidates=passed,
        pool=pool.drop(columns="passed"),
        elapsed_seconds=time.time() - t0,
        vector_size=vector_size,
        model_tag=f"real (trained weights, run/models/, vector_size={vector_size})",
    )

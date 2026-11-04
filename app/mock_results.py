"""
Mock inference results for the Intelligent Antibodies dashboard.

This module stands in for the real generation pipeline in
`intelligent_antibodies/modules/utils/inference.py` so the Streamlit
interface can be designed and reviewed before the trained model weights are
wired in. It deliberately mirrors that pipeline's actual mechanics rather
than inventing unrelated metrics:

- Candidates are sampled as points `z` in a 2-D latent space via
  ``z = temperature * N(0, 1)``, exactly as
  `inference.generate_antibody_sequence` does (``VAEFull`` hard-codes
  ``latent_dim = 2``). The `temperature` knob here is the same parameter
  added to that function.
- Each candidate's `interaction_score` stands in for the sigmoid output of
  `SiameseInteractionClassifier` / `test_interaction` (a 0-1 probability,
  thresholded the same way).
- `decode_confidence` stands in for how "sure" the VAE decoder's one-hot
  reconstruction is at each position before `ProteinOneHotEncoder.decode`
  takes the argmax -- samples drawn further from the training prior (large
  `temperature`, large ``||z||``) are modelled as decoding less confidently,
  which is the real trade-off a sampling-temperature knob controls.
- Sequence length is capped at the caller's `vector_size` (default
  `VECTOR_SIZE`, 200), the same encoding window `VAEFull` /
  `main_pipeline.py` are built around.

Swapping `generate_candidates()` for a real call into
`get_unique_interacting_antibodies()` (now temperature-aware) is the
intended integration point once trained weights are available -- the rest
of the UI only consumes the DataFrame/`GenerationRun` shape below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from intelligent_antibodies.modules.utils.encoding import AMINO_ACID_ALPHABET as _REAL_ALPHABET

# The 20 standard amino acids used to synthesize mock sequences -- matches
# intelligent_antibodies.modules.utils.encoding.AMINO_ACID_ALPHABET minus the
# non-standard X/U placeholders used there only for padding.
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

# Rough natural abundance weighting (Swiss-Prot averages) so generated
# sequences look like plausible CDR loops rather than uniform noise.
_AA_WEIGHTS = np.array([
    8.25, 5.53, 4.06, 5.46, 1.37, 3.93, 6.72, 7.07, 2.27, 5.96,
    9.66, 5.84, 2.42, 3.86, 4.70, 6.56, 5.34, 1.08, 2.92, 6.87,
])
_AA_WEIGHTS = _AA_WEIGHTS / _AA_WEIGHTS.sum()

# From intelligent_antibodies/modules/models/VAEFull.py / main_pipeline.py.
# ALPHABET_SIZE is derived from the real encoder (not hardcoded) -- an
# earlier hardcoded "18" here (and in main_pipeline.py) had drifted out of
# sync with the encoder's actual 22-letter alphabet; see encoding.py.
VECTOR_SIZE = 200
ALPHABET_SIZE = len(_REAL_ALPHABET)
LATENT_DIM = 2

TIER_THRESHOLDS = {"Lead candidate": 0.95, "High confidence": 0.85}

# --------------------------------------------------------------------------
# Real, published per-residue scales used for the sequence-analysis panel
# and the schematic secondary-structure / "fold" view. These are standard
# bioinformatics tables (not fabricated), used here for a simplified,
# clearly-labelled heuristic rather than an actual structure prediction.
# --------------------------------------------------------------------------

# Kyte & Doolittle (1982) hydrophobicity scale.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Simplified physicochemical class, used for the colorized sequence viewer.
RESIDUE_CLASS = {
    "K": "Positive", "R": "Positive", "H": "Positive",
    "D": "Negative", "E": "Negative",
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar", "Y": "Polar", "C": "Polar",
    "A": "Hydrophobic", "V": "Hydrophobic", "L": "Hydrophobic", "I": "Hydrophobic",
    "M": "Hydrophobic", "F": "Hydrophobic", "W": "Hydrophobic",
    "G": "Special", "P": "Special",
}
RESIDUE_CLASS_ORDER = ["Positive", "Hydrophobic", "Polar", "Special", "Negative"]

# Rendering fallback for letters the five classes above don't cover -- kept out
# of RESIDUE_CLASS_ORDER on purpose, since that list drives the profile's
# percentage columns and the sequence viewer's legend.
RESIDUE_CLASS_UNKNOWN = "Non-standard"
RESIDUE_CLASS_COLOR = {
    "Positive": "#2a78d6",      # categorical slot 1
    "Hydrophobic": "#eb6834",   # categorical slot 2
    "Polar": "#1baf7a",         # categorical slot 3
    "Special": "#eda100",       # categorical slot 4
    "Negative": "#e87ba4",      # categorical slot 5
    RESIDUE_CLASS_UNKNOWN: "#898781",   # grey -- outside the 20 standard AAs
}

# Approximate Chou & Fasman (1978) conformational propensities. Used only to
# drive a simplified, illustrative secondary-structure call -- NOT a real
# structure-prediction model (no DSSP/AlphaFold/ESMFold involved).
CHOU_FASMAN_HELIX = {
    "E": 1.51, "M": 1.45, "A": 1.42, "L": 1.21, "K": 1.16, "F": 1.13, "Q": 1.11,
    "W": 1.08, "I": 1.08, "V": 1.06, "D": 1.01, "H": 1.00, "R": 0.98, "T": 0.83,
    "S": 0.77, "C": 0.70, "Y": 0.69, "N": 0.67, "P": 0.57, "G": 0.57,
}
CHOU_FASMAN_SHEET = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "C": 1.19, "T": 1.19,
    "L": 1.30, "Q": 1.10, "M": 1.05, "R": 0.93, "H": 0.87, "N": 0.89, "K": 0.74,
    "S": 0.75, "G": 0.75, "A": 0.83, "D": 0.54, "P": 0.55, "E": 0.37,
}

SS_COLOR = {"Helix": "#e34948", "Sheet": "#eda100", "Coil": "#898781"}
SS_LABEL = {"H": "Helix", "E": "Sheet", "C": "Coil"}


# --------------------------------------------------------------------------
# Sequence sanitizing.
#
# The real encoder's alphabet (encoding.AMINO_ACID_ALPHABET) has 22 letters:
# the 20 standard amino acids plus 'X' (its padding / unknown placeholder) and
# 'U' (selenocysteine). Two consequences the panels above have to survive:
#
# - A real VAE decode is an argmax over all 22 letters at every one of the
#   `vector_size` positions, so decoded sequences legitimately contain X and U.
#   The per-residue tables above are keyed on the 20 standard residues, so
#   feeding a raw decode straight to `physicochemical_profile` raised
#   `KeyError: 'X'`. `clean_generated_sequence` is what real_pipeline runs
#   every decode through before it reaches a DataFrame or the UI.
# - A user-supplied antigen (typed, pasted, or read from a FASTA file) can
#   carry lowercase letters, gaps, '*', line breaks or ambiguity codes, none of
#   which `ProteinOneHotEncoder.encode` knows -- it would raise `KeyError` too.
#   `sanitize_antigen_sequence` maps those onto the alphabet instead.
# --------------------------------------------------------------------------

def sanitize_antigen_sequence(sequence: str) -> str:
    """
    Coerce a user-supplied sequence into the real encoder's 22-letter alphabet.

    Uppercases; drops anything that isn't a letter (whitespace, digits, gaps
    such as ``-``/``.``, stop codons ``*``); and folds any remaining unknown
    letter -- the ambiguity codes B/Z/J/O, for instance -- onto ``X``, the
    alphabet's own unknown placeholder, which preserves the residue's position
    rather than silently shortening the sequence.
    """
    return "".join(
        char if char in _REAL_ALPHABET else "X"
        for char in sequence.upper()
        if char.isalpha()
    )


def clean_generated_sequence(sequence: str) -> str:
    """
    Turn a raw VAE decode into a sequence of standard amino acids only.

    ``U`` (selenocysteine) is folded onto ``C``, exactly as
    `encoding.NLFEncoder` / `encoding.BLOSUMEncoder` already do. ``X`` is the
    encoder's padding placeholder -- a decoded position the model is really
    saying is empty -- so it is dropped rather than substituted, which also
    trims the padding tail off a decode shorter than the encoding window. The
    result may be empty if a decode is all padding; callers skip those.
    """
    return "".join(aa for aa in sequence.upper().replace("U", "C") if aa in AMINO_ACIDS)


@dataclass
class GenerationRun:
    """A single mock generation call: its parameters, results, and pool."""

    antigen_sequence: str
    n_requested: int
    threshold: float
    temperature: float
    seed: int
    candidates: pd.DataFrame      # the n_requested (or fewer) that passed
    pool: pd.DataFrame            # every sampled point, pass or fail
    elapsed_seconds: float
    vector_size: int = VECTOR_SIZE   # encoding window / requested sequence length
    model_tag: str = "mock-generator-v0 (VAE+Siamese not yet connected)"
    timestamp: float = field(default_factory=time.time)
    batches_run: int = 1             # sampling rounds spent reaching n_requested
    target_reached: bool = True      # False if the sampling budget ran out first
    # What the trained classifier's scores actually look like on held-out
    # pairs (written by scripts/train_siamese.py; None for a mock run). The
    # UI needs it to explain an empty result: a classifier whose scores top
    # out at 0.69 can never clear a 0.80 threshold, however long it samples.
    calibration: Optional[dict] = None

    @property
    def n_passed(self) -> int:
        return len(self.candidates)

    @property
    def n_pool_generated(self) -> int:
        return len(self.pool)

    @property
    def best_score(self) -> float:
        """Highest score in the whole sampled pool, passing or not."""
        if self.pool.empty or "interaction_score" not in self.pool:
            return float("nan")
        return float(self.pool["interaction_score"].max())


def _random_sequence(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list(AMINO_ACIDS), size=length, p=_AA_WEIGHTS))


def position_confidence_trace(base_confidence: float, length: int, seed: int = 0) -> List[float]:
    """
    Per-residue decode confidence, shaped around a known scalar confidence.

    For mock candidates this stands in for the VAE decoder's one-hot
    activations before `ProteinOneHotEncoder.decode`'s argmax. For *real*
    candidates (see `app/real_pipeline.py`), `base_confidence` is the real
    `decode_confidence` returned by `inference.generate_antibody_sequence` --
    only the per-position breakdown shape is synthesized, not the scalar.
    """
    rng = np.random.default_rng(seed)
    trace = base_confidence + rng.normal(0, 0.07, size=length)
    # A handful of locally harder-to-decode positions (e.g. loop regions).
    dip_positions = rng.choice(length, size=max(1, length // 12), replace=False)
    trace[dip_positions] -= rng.uniform(0.1, 0.3, size=len(dip_positions))
    return np.clip(trace, 0.15, 0.99).round(3).tolist()


def _amino_acid_composition(sequence: str) -> pd.DataFrame:
    """Normalized amino-acid frequency table for a single sequence."""
    counts = pd.Series(list(sequence)).value_counts(normalize=True) * 100
    counts = counts.reindex(sorted(AMINO_ACIDS)).fillna(0.0)
    return counts.rename("pct").rename_axis("residue").reset_index()


def tier_for_score(score: float) -> str:
    if score >= TIER_THRESHOLDS["Lead candidate"]:
        return "Lead candidate"
    if score >= TIER_THRESHOLDS["High confidence"]:
        return "High confidence"
    return "Candidate"


def _sample_round(
    rng: np.random.Generator,
    center: np.ndarray,
    size: int,
    temperature: float,
    threshold: float,
    lo: int,
    hi: int,
) -> pd.DataFrame:
    """
    One round of simulated latent sampling + scoring (see `generate_candidates`).

    Split out so the rejection-sampling loop can draw as many rounds as it
    takes to reach the requested number of passing candidates.
    """
    # --- Latent-space sampling, exactly as inference.generate_antibody_sequence ---
    z = temperature * rng.normal(size=(size, LATENT_DIM))

    # --- Fake "interaction score", modelled as a smooth function of latent
    # position so the latent-space plot has structure to show: a hidden
    # antigen-favorable region, found via the usual rejection sampling. ---
    dist_to_center = np.linalg.norm(z - center, axis=1)
    score_logit = 4.0 - 2.6 * dist_to_center + rng.normal(0, 0.7, size=size)
    scores = np.clip(1 / (1 + np.exp(-score_logit)), 0.01, 0.995)

    # --- Fake decode confidence: decays with distance from the prior's
    # mode, i.e. with how far temperature pushed the sample into the tails. ---
    dist_from_origin = np.linalg.norm(z, axis=1)
    decode_confidence = np.clip(
        0.96 - 0.11 * dist_from_origin + rng.normal(0, 0.04, size=size), 0.15, 0.99
    )

    lengths = rng.integers(max(lo, 1), hi + 1, size=size)
    sequences = [_random_sequence(rng, int(length)) for length in lengths]

    return pd.DataFrame({
        "z1": z[:, 0],
        "z2": z[:, 1],
        "sequence": sequences,
        "length": lengths,
        "interaction_score": scores,
        "decode_confidence": decode_confidence,
        "passed": scores >= threshold,
    })


def generate_candidates(
    antigen_sequence: str,
    n_requested: int = 20,
    threshold: float = 0.80,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    vector_size: int = VECTOR_SIZE,
    max_batches: int = 40,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> GenerationRun:
    """
    Simulate a batch antibody-generation run against a given antigen.

    Mirrors the real rejection-sampling loop in
    `inference.get_unique_interacting_antibodies`: latent points are sampled at
    the given `temperature`, each is "decoded" into a sequence and scored for
    predicted interaction with the antigen, and only candidates clearing
    `threshold` are kept. Sampling keeps going round after round until
    `n_requested` candidates have passed (or `max_batches` rounds have been
    spent), so `n_requested` is a target to reach rather than just a cap --
    matching `real_pipeline.generate_candidates_real`.

    Parameters
    ----------
    antigen_sequence : str
        The target antigen sequence (used only to size/vary the mock output;
        not actually encoded by a real model here).
    n_requested : int
        Maximum number of passing candidates to return.
    threshold : float
        Minimum interaction score (0-1) for a candidate to be kept, as in
        `inference.test_interaction`.
    temperature : float
        Latent-sampling temperature: ``z = temperature * N(0, 1)``, matching
        the parameter added to `inference.generate_antibody_sequence`.
    seed : int, optional
        RNG seed. A fresh run should omit this so results vary per click.
    min_length, max_length : int, optional
        Bounds on simulated antibody sequence length. Omitted (the default),
        they scale with `vector_size`; either way they are clamped to it.
    vector_size : int
        The one-hot encoding window the real models would use, i.e. the
        requested generated-sequence length. Simulated sequences never
        exceed it, since the real encoder truncates at that width.
    max_batches : int
        Safety budget: at most this many sampling rounds ("attempts") are
        spent looking for `n_requested` passing candidates. A strict
        `threshold` can be unreachable, so the loop must be bounded;
        `GenerationRun.target_reached` says whether the target was actually
        met. Named to match `real_pipeline.generate_candidates_real`'s
        `max_batches`, so a caller (e.g. the dashboard's "max attempts"
        control) can pass the same value to either generator.
    progress : callable, optional
        Called as ``progress(n_passed, n_requested, pool_size)`` after every
        round, so a caller can report real progress towards the target.

    Returns
    -------
    GenerationRun
    """
    t0 = time.time()
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)

    # Default bounds track the encoding window: the 0.52-0.68 ratios reproduce
    # the previous fixed 105-135 aa range at the default 200 aa window (roughly
    # a real antibody chain), and stay proportionate for any other window.
    hi = min(max_length if max_length is not None else round(0.68 * vector_size), vector_size)
    lo = min(min_length if min_length is not None else round(0.52 * vector_size), hi)

    # The hidden antigen-favorable region of latent space. Drawn once, outside
    # the sampling loop, so every round samples against the same "truth".
    center = rng.normal(0, 0.5, size=LATENT_DIM)

    round_size = max(80, n_requested * 8)

    # Rejection sampling towards the target: keep drawing rounds until
    # n_requested candidates have passed, bounded by max_batches.
    rounds: List[pd.DataFrame] = []
    n_found = 0
    rounds_run = 0
    # `not rounds` keeps a degenerate n_requested <= 0 from leaving the pool
    # (and therefore every downstream plot) empty.
    while not rounds or (n_found < n_requested and rounds_run < max_batches):
        chunk = _sample_round(rng, center, round_size, temperature, threshold, lo, hi)
        rounds.append(chunk)
        rounds_run += 1
        n_found += int(chunk["passed"].sum())
        if progress is not None:
            progress(min(n_found, n_requested), n_requested, round_size * rounds_run)

    pool = pd.concat(rounds, ignore_index=True)
    pool_size = len(pool)

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
        position_confidence_trace(conf, int(length), seed=rng.integers(0, 2**32 - 1))
        for conf, length in zip(passed["decode_confidence"], passed["length"])
    ]
    # A strict threshold can exhaust the sampling budget with nothing passing;
    # an empty frame has no columns to read, so the columns are declared
    # explicitly (same guard as real_pipeline.generate_candidates_real).
    if not passed.empty:
        profiles = pd.DataFrame([physicochemical_profile(seq) for seq in passed["sequence"]])
        passed["hydrophobicity"] = profiles["hydrophobicity"]
        passed["net_charge"] = profiles["net_charge"]
    else:
        passed["hydrophobicity"] = pd.Series(dtype=float)
        passed["net_charge"] = pd.Series(dtype=int)

    # Simulated compute time, scaling gently with how much sampling it took.
    elapsed = 0.8 + pool_size / 400 + rng.uniform(-0.2, 0.4)

    return GenerationRun(
        antigen_sequence=antigen_sequence,
        n_requested=n_requested,
        threshold=threshold,
        temperature=temperature,
        seed=int(seed),
        candidates=passed,
        pool=pool,
        elapsed_seconds=max(elapsed, 0.3),
        vector_size=vector_size,
        model_tag=f"mock-generator-v0 (VAE+Siamese not yet connected, vector_size={vector_size})",
        batches_run=rounds_run,
        target_reached=len(passed) >= n_requested,
    )


def amino_acid_composition(sequence: str) -> pd.DataFrame:
    """Public wrapper used by the UI to plot a candidate's AA composition."""
    return _amino_acid_composition(sequence)


def physicochemical_profile(sequence: str) -> dict:
    """
    Real, simple per-sequence properties (not model output): mean Kyte-Doolittle
    hydrophobicity and an approximate net charge at physiological pH
    (``+1`` per K/R/H, ``-1`` per D/E -- a common coarse estimate).

    Only standard residues are scored: the input is run through
    `clean_generated_sequence` first, so a raw VAE decode containing the
    encoder's X/U letters is handled instead of raising `KeyError`. A sequence
    with no scorable residue at all yields a neutral, all-zero profile.
    """
    residues = clean_generated_sequence(sequence)
    class_pct = {cls: 0.0 for cls in RESIDUE_CLASS_ORDER}
    if not residues:
        return {"hydrophobicity": 0.0, "net_charge": 0, **class_pct}

    hydrophobicity = np.mean([KYTE_DOOLITTLE[aa] for aa in residues])
    net_charge = sum(1 for aa in residues if aa in "KRH") - sum(1 for aa in residues if aa in "DE")
    class_counts = pd.Series([RESIDUE_CLASS[aa] for aa in residues]).value_counts(normalize=True) * 100
    class_pct.update({cls: float(class_counts.get(cls, 0.0)) for cls in RESIDUE_CLASS_ORDER})
    return {"hydrophobicity": float(hydrophobicity), "net_charge": int(net_charge), **class_pct}


def predict_secondary_structure(sequence: str, window: int = 4, min_run: int = 3) -> List[str]:
    """
    Simplified Chou-Fasman-style secondary-structure call per residue
    (returns 'H'/'E'/'C'). Deliberately approximate -- intended only to drive
    an illustrative schematic "fold" view, not a real structure prediction.

    One label per input residue, so the result stays aligned with `sequence`
    for the viewer: a non-standard letter (X/U from the encoder's alphabet, or
    anything else) is given a neutral propensity of 1.0 rather than dropped.
    """
    n = len(sequence)
    half = window // 2
    helix_scores = np.array([CHOU_FASMAN_HELIX.get(aa, 1.0) for aa in sequence])
    sheet_scores = np.array([CHOU_FASMAN_SHEET.get(aa, 1.0) for aa in sequence])

    labels = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        h_avg = helix_scores[lo:hi].mean()
        s_avg = sheet_scores[lo:hi].mean()
        if h_avg > 1.03 and h_avg >= s_avg:
            labels.append("H")
        elif s_avg > 1.05 and s_avg > h_avg:
            labels.append("E")
        else:
            labels.append("C")

    # Smooth out runs shorter than min_run (avoid single-residue flicker).
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        if labels[i] != "C" and (j - i) < min_run:
            for k in range(i, j):
                labels[k] = "C"
        i = j
    return labels


def schematic_backbone(sequence: str, ss: Optional[List[str]] = None, seed: int = 0) -> pd.DataFrame:
    """
    A schematic, illustrative 3-D backbone trace coloured by predicted
    secondary structure -- NOT a physically accurate fold and NOT the output
    of a structure-prediction model (no AlphaFold/ESMFold is connected). It
    is only meant to give a visual stand-in until real structure prediction
    is wired in: helical segments spiral, sheet segments zigzag, coil
    segments wander, consistent with `predict_secondary_structure`.
    """
    if ss is None:
        ss = predict_secondary_structure(sequence)
    rng = np.random.default_rng(seed)

    n = len(sequence)
    coords = np.zeros((n, 3))
    x = y = z = 0.0
    helix_angle = 0.0
    sheet_phase = 0.0
    direction = 1.0

    for i in range(n):
        label = ss[i]
        if label == "H":
            helix_angle += np.deg2rad(100)
            y = 2.3 * np.cos(helix_angle)
            z = 2.3 * np.sin(helix_angle)
            x += 1.5
        elif label == "E":
            sheet_phase += np.pi
            y = 2.0 * direction * (1 if (i // 2) % 2 == 0 else -1)
            z += rng.normal(0, 0.15)
            x += 3.4
        else:
            y += rng.normal(0, 0.9)
            z += rng.normal(0, 0.9)
            x += 3.2
        coords[i] = [x, y, z]

    return pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "residue": list(sequence), "position": np.arange(1, n + 1),
        "ss": ss, "ss_label": [SS_LABEL[s] for s in ss],
    })


def _residue_color(aa: str) -> str:
    """Viewer colour for a residue; non-standard letters render grey."""
    return RESIDUE_CLASS_COLOR[RESIDUE_CLASS.get(aa, RESIDUE_CLASS_UNKNOWN)]


def colorize_sequence_html(sequence: str, width: int = 60) -> str:
    """
    Render a sequence as monospace HTML colored by physicochemical residue
    class (see RESIDUE_CLASS), chunked into fixed-width rows with a position
    ruler -- a standard colored sequence-viewer layout. Every input character
    is rendered, non-standard ones in grey, so the ruler stays truthful.
    """
    rows = []
    for start in range(0, len(sequence), width):
        chunk = sequence[start:start + width]
        spans = "".join(
            f'<span style="background:{_residue_color(aa)}22;'
            f'color:{_residue_color(aa)};'
            f'font-weight:600;padding:1px 0;">{aa}</span>'
            for aa in chunk
        )
        rows.append(
            f'<div style="white-space:nowrap;">'
            f'<span style="color:#898781;display:inline-block;width:3.5em;">{start + 1}</span>{spans}'
            f"</div>"
        )
    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;'
        'font-size:0.85rem;line-height:1.6;overflow-x:auto;">' + "".join(rows) + "</div>"
    )


if __name__ == "__main__":
    # Self-check: generated lengths must honour the requested window, including
    # windows smaller than the old fixed 105-135 aa range.
    for vs in (30, 200, 450):
        run = generate_candidates("ACDEFGHIKL", n_requested=10, seed=1, vector_size=vs)
        assert run.pool["length"].max() <= vs, (vs, run.pool["length"].max())
        assert run.pool["length"].min() >= 1
        assert run.pool["sequence"].map(len).equals(run.pool["length"]), vs
        assert f"vector_size={vs}" in run.model_tag
        assert run.vector_size == vs
    # Explicit bounds are still clamped to the window.
    run = generate_candidates("ACDEFGHIKL", seed=1, vector_size=50, min_length=105, max_length=135)
    assert run.pool["length"].max() <= 50

    # Sampling runs until the requested number of candidates has passed...
    run = generate_candidates("ACDEFGHIKL", n_requested=45, threshold=0.93, seed=2)
    assert run.n_passed == 45 and run.target_reached, (run.n_passed, run.target_reached)
    # ...but an unreachable threshold exhausts the budget instead of hanging.
    run = generate_candidates("ACDEFGHIKL", n_requested=50, threshold=0.995, seed=2, max_batches=2)
    assert not run.target_reached and run.batches_run == 2
    assert {"hydrophobicity", "net_charge"} <= set(run.candidates.columns)

    # Raw VAE decodes carry the encoder's X/U letters (see clean_generated_sequence);
    # the per-residue panels must characterize them instead of raising KeyError.
    raw_decode = "ARNDCUKRH" + "X" * 30
    assert clean_generated_sequence(raw_decode) == "ARNDCCKRH"
    profile = physicochemical_profile(raw_decode)
    assert profile["net_charge"] == 3, profile   # R/K/R/H minus D
    assert physicochemical_profile("XXXX")["hydrophobicity"] == 0.0
    assert len(predict_secondary_structure(raw_decode)) == len(raw_decode)
    assert len(schematic_backbone(raw_decode)) == len(raw_decode)
    assert colorize_sequence_html(raw_decode)

    # A user-supplied antigen is coerced onto the encoder's 22-letter alphabet.
    assert sanitize_antigen_sequence("mkv fgr-cel*aaa\n") == "MKVFGRCELAAA"
    assert sanitize_antigen_sequence("ABZJO") == "AXXXX"   # ambiguity codes -> X
    assert set(sanitize_antigen_sequence("acdefghiklZ")) <= set(_REAL_ALPHABET)

    print("mock_results self-check OK")

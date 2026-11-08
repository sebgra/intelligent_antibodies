"""
What the two temperature knobs actually do.

`temperature` widens the latent draw; `residue_temperature` samples each
position from the decoder's distribution instead of taking the argmax. Both
are advertised as "diversity" controls, so the question is what each costs.

    uv run python scripts/figures/sampling_temperature.py
"""
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (AQUA, BLUE, CRITICAL, INK, INK_2, INK_MUTED, ORANGE, SEQ,
                     VECTOR_SIZE, background_frequencies, build_pssm, finish,
                     load_models_and_stats, pssm_score, real_chains,
                     sampled_candidates, style, tidy)
from intelligent_antibodies.modules.utils.chains import classify_chain
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.inference import test_interaction
from intelligent_antibodies.modules.utils.paths import siamese_calibration_path

style()
models, stats = load_models_and_stats()
encoder = ProteinOneHotEncoder()
calibration = json.loads(siamese_calibration_path(VECTOR_SIZE).read_text())
CUT = float(calibration["selection_threshold"])

chains = real_chains()
profile = build_pssm(chains["sequence"].tolist())
background = background_frequencies(chains["sequence"].tolist())

ANTIGEN = ("KVFGRCELAAAMKRHGLANYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRT"
           "PGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")
onehot_antigen = encoder.encode(pd.Series([ANTIGEN]), VECTOR_SIZE)

N = 120


def identity(batch, cap=600):
    cleaned = [b.replace("X", "").replace("U", "") for b in batch]
    values = []
    for a, b in itertools.islice(itertools.combinations(cleaned, 2), cap):
        n = min(len(a), len(b))
        if n:
            values.append(100 * sum(x == y for x, y in zip(a[:n], b[:n])) / n)
    return float(np.mean(values)) if values else float("nan")


def profile_batch(batch, seed):
    """Every number this figure plots, for one batch of candidates."""
    cleaned = [s.replace("X", "").replace("U", "") for s in batch]
    scores = []
    for seq in cleaned:
        onehot = encoder.encode([seq], VECTOR_SIZE)
        _, score = test_interaction(onehot, onehot_antigen, models["siamese"])
        scores.append(score)
    scores = np.array(scores)
    kinds = [classify_chain(s) for s in cleaned]
    return {
        "identity": identity(batch),
        "plausibility": float(np.median([pssm_score(s, profile, background)
                                         for s in cleaned])),
        "pass_rate": 100 * float((scores >= CUT).mean()),
        "classifiable": 100 * float(np.mean([k is not None for k in kinds])),
        "length": float(np.median([len(s) for s in cleaned])),
    }


sweeps = {}
LATENT_TEMPS = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
RESIDUE_TEMPS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

for i, t in enumerate(LATENT_TEMPS):
    batch = sampled_candidates(n=N, temperature=t, seed=100 + i,
                               models=models, stats=stats)
    sweeps[("latent", t)] = profile_batch(batch, 100 + i)
    print(f"  latent T={t}: {sweeps[('latent', t)]}")

for i, t in enumerate(RESIDUE_TEMPS):
    batch = sampled_candidates(n=N, residue_temperature=t, seed=200 + i,
                               models=models, stats=stats)
    sweeps[("residue", t)] = profile_batch(batch, 200 + i)
    print(f"  residue T={t}: {sweeps[('residue', t)]}")

METRICS = [
    ("identity", "Mean pairwise identity (%)", BLUE, "lower = more diverse"),
    ("plausibility", "Median plausibility (log-odds)", ORANGE, "higher = more antibody-like"),
    ("pass_rate", f"Candidates above {CUT:.2f} (%)", AQUA, "the affinity filter"),
    ("classifiable", "Recognisable chain type (%)", SEQ[9], "did the fold come out intact"),
]

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex="row")
for row, (kind, temps, label) in enumerate((
        ("latent", LATENT_TEMPS, "Latent temperature  (width of the z draw)"),
        ("residue", RESIDUE_TEMPS, "Residue temperature  (0 = greedy argmax)"))):
    for col, (key, ylabel, colour, hint) in enumerate(METRICS):
        ax = axes[row][col]
        values = [sweeps[(kind, t)][key] for t in temps]
        ax.plot(temps, values, color=colour, marker="o", markersize=7)
        baseline = sweeps[(kind, temps[0] if kind == "residue" else 1.0)][key]
        ax.axhline(baseline, color=INK_MUTED, linestyle=":", linewidth=1.3)
        tidy(ax, label if col == 1 else "", ylabel)
        ax.set_title(hint, fontsize=9.5, color=INK_2, pad=6)
        if key in ("identity", "pass_rate", "classifiable"):
            ax.set_ylim(0, max(102, max(values) * 1.15))

for row, label in enumerate(("Latent temperature", "Residue temperature")):
    axes[row][0].annotate(label, (-0.42, 0.5), xycoords="axes fraction",
                          rotation=90, va="center", ha="center",
                          fontsize=11, color=INK, fontweight="600")

fig.suptitle("The two temperature knobs buy diversity at different prices",
             fontsize=12.5, y=1.03)
finish(fig, "25_temperature_effects.png",
       f"{N} candidates per setting, scored against hen egg-white lysozyme. Dotted lines mark "
       "the defaults (latent temperature 1.0, greedy residues). Both knobs buy diversity by "
       "spending plausibility and fold integrity. Note the third column: the share of "
       "candidates passing the affinity filter RISES as the sequences get less antibody-like, "
       "which says more about the classifier than about the candidates.")

# A second, blunter view: what the sequences themselves look like.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, (kind, temps, label) in zip(axes, (
        ("latent", LATENT_TEMPS, "Latent temperature"),
        ("residue", RESIDUE_TEMPS, "Residue temperature"))):
    diversity = [100 - sweeps[(kind, t)]["identity"] for t in temps]
    intact = [sweeps[(kind, t)]["classifiable"] for t in temps]
    ax.plot(temps, diversity, color=BLUE, marker="o", markersize=7,
            label="Diversity (100 − pairwise identity)")
    ax.plot(temps, intact, color=CRITICAL, marker="s", markersize=7,
            label="Chains with a recognisable fold")
    tidy(ax, label, "Percent")
    ax.set_ylim(0, 105)
    ax.set_title(label, fontsize=10.5, pad=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, fontsize=9.5, loc="lower center",
           bbox_to_anchor=(0.5, 1.005), frameon=False)
fig.suptitle("Diversity against integrity", fontsize=12.5, y=1.13)
finish(fig, "26_temperature_tradeoff.png",
       "Both knobs trade structure for diversity; the residue knob does it roughly twice as "
       "steeply, which is why it defaults to off. 'Recognisable fold' means the decode still "
       "carries a framework motif identifying it as a heavy or a light chain.")
print("temperature figures done")

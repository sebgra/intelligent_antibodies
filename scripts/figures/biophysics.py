"""
Does a generated chain have the *architecture* of a real one?

Three structural checks that do not need a folding model, each comparing
generated candidates against the real SAbDab chains:

  * where the two disulfide-forming cysteines sit, and how far apart,
  * where the variable domain ends,
  * how plausible each position looks under a profile built from real chains.

    uv run python scripts/figures/biophysics.py
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import (AQUA, BLUE, CRITICAL, INK, INK_2, INK_MUTED, ORANGE, SEQ,
                     background_frequencies, build_pssm, cysteine_anchors,
                     finish, legend_above, pssm_score, real_chains,
                     sampled_candidates, style, tidy, variable_domain_end)
from intelligent_antibodies.modules.utils.chains import classify_chain

style()
chains = real_chains()
real = {kind: [s for s, c in zip(chains["sequence"], chains["chain"]) if c == kind]
        for kind in ("heavy", "light")}

generated_all = sampled_candidates(n=600)
generated = {kind: [s for s in (sampled_candidates(n=400, chain_type=kind, seed=1))
                    if classify_chain(s) == kind]
             for kind in ("heavy", "light")}
for kind in ("heavy", "light"):
    print(f"{kind}: {len(real[kind])} real, {len(generated[kind])} generated")

REAL_STYLE = dict(color=ORANGE, alpha=0.8)
GEN_STYLE = dict(color=BLUE, alpha=0.65)


def paired_hist(ax, real_values, gen_values, bins, xlabel, title):
    """
    Real and generated on the same axes, as PERCENTAGES.

    Raw counts would be misleading here: there are roughly twice as many real
    chains as generated ones, so the real histogram simply towers over the
    other whatever the shapes are.
    """
    weights_r = np.ones(len(real_values)) / max(len(real_values), 1) * 100
    weights_g = np.ones(len(gen_values)) / max(len(gen_values), 1) * 100
    ax.hist(real_values, bins=bins, weights=weights_r,
            label=f"real (n={len(real_values)})", **REAL_STYLE)
    ax.hist(gen_values, bins=bins, weights=weights_g,
            label=f"generated (n={len(gen_values)})", **GEN_STYLE)
    for values, colour in ((real_values, ORANGE), (gen_values, BLUE)):
        if len(values):
            ax.axvline(np.median(values), color=colour, linestyle="--", linewidth=1.5)
    # Clip the view to where the data actually is; these distributions are
    # very tight and a fixed wide range leaves the panel mostly empty.
    combined = list(real_values) + list(gen_values)
    if combined:
        low, high = np.percentile(combined, [0.5, 99.5])
        pad = max(2.0, 0.12 * (high - low))
        ax.set_xlim(low - pad, high + pad)
    tidy(ax, xlabel, "Chains (%)")
    ax.set_title(title, fontsize=10.5, pad=8)


# ====================================== disulfide anchors (tasks 3 and 4)
anchors = {}
for source, pool in (("real", real), ("generated", generated)):
    for kind in ("heavy", "light"):
        found = [cysteine_anchors(s) for s in pool[kind]]
        found = [f for f in found if f]
        anchors[(source, kind)] = found
        rate = 100 * len(found) / max(len(pool[kind]), 1)
        print(f"  {source:9} {kind:5}: both cysteines found in {rate:.0f}% of chains")

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
for row, kind in enumerate(("heavy", "light")):
    r = anchors[("real", kind)]
    g = anchors[("generated", kind)]
    paired_hist(axes[row][0], [a for a, _ in r], [a for a, _ in g],
                np.arange(14, 38, 1), "Position of the first cysteine",
                f"{kind.capitalize()} — framework-1 cysteine")
    paired_hist(axes[row][1], [b for _, b in r], [b for _, b in g],
                np.arange(60, 115, 2), "Position of the second cysteine",
                f"{kind.capitalize()} — framework-3 cysteine")
    paired_hist(axes[row][2], [b - a for a, b in r], [b - a for a, b in g],
                np.arange(45, 95, 2), "Residues between the two cysteines",
                f"{kind.capitalize()} — disulfide loop span")
# One legend for the whole figure: three per-panel legends collided with the
# panel titles.
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, fontsize=9.5, loc="lower center",
           bbox_to_anchor=(0.5, 1.005), frameon=False)
fig.suptitle("The internal disulfide: both cysteines, and the loop they close",
             fontsize=12.5, y=1.08)
finish(fig, "22_disulfide_anchors.png",
       "Every immunoglobulin variable domain is pinned by one disulfide bond, from a cysteine "
       "near the end of framework 1 to its partner in framework 3. Dashed lines mark medians.")

# ====================================== where the variable domain ends (task 3)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, kind in zip(axes, ("light", "heavy")):
    r = [variable_domain_end(s) for s in real[kind]]
    g = [variable_domain_end(s) for s in generated[kind]]
    r = [x for x in r if x is not None]
    g = [x for x in g if x is not None]
    paired_hist(ax, r, g, np.arange(84, 134, 2),
                "Position where framework 4 begins",
                f"{kind.capitalize()} chains")
    found_real = 100 * len(r) / max(len(real[kind]), 1)
    found_gen = 100 * len(g) / max(len(generated[kind]), 1)
    ax.annotate(f"domain closes in {found_real:.0f}% of real,\n{found_gen:.0f}% of generated",
                (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5, color=INK_2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, fontsize=9.5, loc="lower center",
           bbox_to_anchor=(0.5, 1.005), frameon=False)
fig.suptitle("Where the variable domain closes", fontsize=12.5, y=1.13)
finish(fig, "23_variable_domain_extent.png",
       "Framework 4 is the last stretch of the variable domain, recognisable by its WGxG "
       "(heavy) or FGxG (light) motif. A domain that never closes is a decode that ran past "
       "the end of the fold without finishing it.")

# ====================================== plausibility (task 5)
# Profile built on a held-out half of the real chains, so a real sequence is
# not scored against a profile that contains it.
rng = np.random.default_rng(0)
scores = {}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, kind in zip(axes, ("heavy", "light")):
    pool = real[kind]
    order = rng.permutation(len(pool))
    train = [pool[i] for i in order[: len(pool) // 2]]
    held_out = [pool[i] for i in order[len(pool) // 2:]]
    profile = build_pssm(train)
    background = background_frequencies(train)

    real_scores = [pssm_score(s, profile, background) for s in held_out]
    gen_scores = [pssm_score(s, profile, background) for s in generated[kind]]
    shuffled = ["".join(rng.permutation(list(s))) for s in held_out[:200]]
    shuffled_scores = [pssm_score(s, profile, background) for s in shuffled]
    scores[kind] = (real_scores, gen_scores, shuffled_scores)

    bins = np.linspace(-1.5, 3.0, 46)
    for values, colour, alpha, label in (
        (real_scores, ORANGE, 0.8, f"real, held out (n={len(real_scores)})"),
        (gen_scores, BLUE, 0.65, f"generated (n={len(gen_scores)})"),
        (shuffled_scores, SEQ[1], 0.85, f"shuffled real (n={len(shuffled_scores)})"),
    ):
        ax.hist(values, bins=bins, color=colour, alpha=alpha, label=label,
                weights=np.ones(len(values)) / max(len(values), 1) * 100)
    for values, colour in ((real_scores, ORANGE), (gen_scores, BLUE),
                           (shuffled_scores, INK_MUTED)):
        ax.axvline(np.median(values), color=colour, linestyle="--", linewidth=1.5)
    tidy(ax, "Mean per-residue log-odds vs background", "Chains (%)")
    ax.set_title(f"{kind.capitalize()} chains", fontsize=10.5, pad=8)
    print(f"  {kind}: real {np.median(real_scores):.2f}, generated "
          f"{np.median(gen_scores):.2f}, shuffled {np.median(shuffled_scores):.2f}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, fontsize=9.5, loc="lower center",
           bbox_to_anchor=(0.5, 1.005), frameon=False)
fig.suptitle("How antibody-like is a generated sequence, position by position?",
             fontsize=12.5, y=1.13)
finish(fig, "24_plausibility.png",
       "Each chain scored against a profile built from one half of the real chains and tested "
       "on the other half, so no sequence is scored against a profile containing it. Shuffling "
       "a real chain destroys its positional structure and gives the floor.")
print("biophysics figures done")

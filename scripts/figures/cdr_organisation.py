"""
Has the generator learned WHERE an antibody is allowed to vary?

Positional agreement across a set of chains is high in the frameworks and low
in the CDR loops. If the model has learned the organisation rather than just
the vocabulary, its candidates should show the same profile as real chains.

    uv run python scripts/figures/cdr_organisation.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import pearsonr

from _common import (AQUA, BLUE, CRITICAL, GRID, INK, INK_2, INK_MUTED, ORANGE,
                     SEQ, SURFACE, finish, real_chains, sampled_candidates,
                     style, tidy)
from intelligent_antibodies.modules.utils.encoding import AMINO_ACID_ALPHABET

style()
rng = np.random.default_rng(0)
N_SHOW, WINDOW = 14, 120

chains = real_chains()
real_pool = [s for s in chains["sequence"] if len(s) >= WINDOW]
generated_pool = [s.replace("X", "").replace("U", "")
                  for s in sampled_candidates(n=600, seed=0)]
generated_pool = [s for s in generated_pool if len(s) >= WINDOW]
print(f"{len(real_pool)} real and {len(generated_pool)} generated chains of >= {WINDOW} aa")


def agreement(sequences, window=WINDOW):
    """Fraction agreeing on the most common residue, position by position."""
    matrix = np.array([[AMINO_ACID_ALPHABET.index(c) for c in s[:window]]
                       for s in sequences])
    return np.array([np.bincount(matrix[:, j], minlength=22).max() / len(sequences)
                     for j in range(window)]) * 100


# Equal-sized samples: agreement depends on how many sequences you compare.
n = min(N_SHOW, len(real_pool), len(generated_pool))
real_sample = [real_pool[i] for i in rng.choice(len(real_pool), n, replace=False)]
gen_sample = [generated_pool[i] for i in rng.choice(len(generated_pool), n, replace=False)]
real_profile, gen_profile = agreement(real_sample), agreement(gen_sample)
correlation, p_value = pearsonr(real_profile, gen_profile)
print(f"profile correlation r={correlation:.2f} (p={p_value:.1e}); "
      f"means real {real_profile.mean():.0f}% generated {gen_profile.mean():.0f}%")

CDR_BANDS = ((26, 35, "CDR1"), (50, 65, "CDR2"), (95, 110, "CDR3"))
cdr_mask = np.zeros(WINDOW, bool)
for start, end, _ in CDR_BANDS:
    cdr_mask[start:end] = True

# ------------------------------------------------- the main figure (15)
CLASS = {**{c: 0 for c in "AVLIMFWGP"}, **{c: 1 for c in "STNQYC"},
         **{c: 2 for c in "KRH"}, **{c: 3 for c in "DE"}}
CLASS_COLOURS = [SEQ[7], SEQ[3], ORANGE, AQUA]
CLASS_NAMES = ["Hydrophobic / special", "Polar", "Positive", "Negative"]

fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1.5], "hspace": 0.45})

ax = axes[0]
ax.imshow(np.array([[CLASS.get(c, 1) for c in s[:WINDOW]] for s in gen_sample]),
          aspect="auto", cmap=ListedColormap(CLASS_COLOURS), interpolation="nearest",
          vmin=0, vmax=3)
ax.set_yticks(range(n), [f"cand {i + 1}" for i in range(n)], fontsize=8)
ax.set_ylabel("Generated candidates")
ax.set_title(f"{n} independently generated candidates, first {WINDOW} residues")
ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(CLASS_COLOURS, CLASS_NAMES)],
          ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06),
          frameon=False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax = axes[1]
for start, end, label in CDR_BANDS:
    ax.axvspan(start, end, color=ORANGE, alpha=0.14, zorder=0)
    ax.annotate(label, ((start + end) / 2, 97), ha="center", fontsize=8.5,
                color=ORANGE, zorder=4)
ax.plot(range(WINDOW), real_profile, color=ORANGE, label=f"{n} real SAbDab antibodies")
ax.plot(range(WINDOW), gen_profile, color=BLUE, label=f"{n} generated candidates")
# The explicit mean lines: a profile is easy to over-read without them.
ax.axhline(real_profile.mean(), color=ORANGE, linestyle="--", linewidth=1.5, zorder=3)
ax.axhline(gen_profile.mean(), color=BLUE, linestyle="--", linewidth=1.5, zorder=3)
# Opposite ends: the two means are only a few points apart, so labelling both
# at the left edge put them on top of each other.
ax.annotate(f"real mean {real_profile.mean():.0f}%", (1, real_profile.mean()),
            textcoords="offset points", xytext=(2, -13), ha="left", fontsize=8.5,
            color=ORANGE, zorder=4)
ax.annotate(f"generated mean {gen_profile.mean():.0f}%", (WINDOW - 2, gen_profile.mean()),
            textcoords="offset points", xytext=(-2, 7), ha="right", fontsize=8.5,
            color=BLUE, zorder=4)
ax.set_ylim(0, 105)
ax.set_xlim(0, WINDOW - 1)
tidy(ax, "Position in the variable domain", "Agreement at each\nposition (%)")
ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.annotate(f"profiles correlate at r = {correlation:.2f}", (0.5, -0.38),
            xycoords="axes fraction", ha="center", fontsize=9.5, color=INK_2)

finish(fig, "15_generated_alignment.png",
       "Both profiles computed over the same number of sequences, since agreement depends on "
       "sample size. Dashed lines are the means.")

# ------------------------------------- digging deeper: region by region (27)
REGIONS = [("FR1", 0, 26), ("CDR1", 26, 35), ("FR2", 35, 50), ("CDR2", 50, 65),
           ("FR3", 65, 95), ("CDR3", 95, 110), ("FR4", 110, WINDOW)]

# Bootstrap over resampled sets so the bars carry an interval rather than
# resting on one lucky draw of 14 sequences.
def region_means(pool, draws=200):
    """Region means plus the framework-minus-CDR gap, per bootstrap draw."""
    out = {name: [] for name, _, _ in REGIONS}
    gaps = []
    for _ in range(draws):
        sample = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        profile = agreement(sample)
        for name, start, end in REGIONS:
            out[name].append(profile[start:end].mean())
        fr = np.mean([profile[a:b].mean() for nm, a, b in REGIONS if nm.startswith("FR")])
        cdr = np.mean([profile[a:b].mean() for nm, a, b in REGIONS if nm.startswith("CDR")])
        gaps.append(fr - cdr)
    summary = {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
               for k, v in out.items()}
    summary["_gap"] = (np.mean(gaps), np.percentile(gaps, 2.5), np.percentile(gaps, 97.5))
    return summary


real_regions = region_means(real_pool)
gen_regions = region_means(generated_pool)

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

ax = axes[0]
x = np.arange(len(REGIONS))
for offset, (data, colour, label) in zip((-0.2, 0.2), (
        (real_regions, ORANGE, "Real"), (gen_regions, BLUE, "Generated"))):
    means = [data[name][0] for name, _, _ in REGIONS]
    lower = [data[name][0] - data[name][1] for name, _, _ in REGIONS]
    upper = [data[name][2] - data[name][0] for name, _, _ in REGIONS]
    ax.bar(x + offset, means, width=0.36, color=colour, label=label,
           yerr=[lower, upper], ecolor=INK_MUTED, capsize=3, error_kw={"linewidth": 1})
ax.set_xticks(x, [name for name, _, _ in REGIONS])
for i, (name, _, _) in enumerate(REGIONS):
    if name.startswith("CDR"):
        ax.axvspan(i - 0.45, i + 0.45, color=ORANGE, alpha=0.10, zorder=0)
tidy(ax, "", "Mean agreement (%)")
ax.set_ylim(0, 100)
ax.set_title("Agreement region by region", pad=8)
ax.legend(fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.09), frameon=False)

ax = axes[1]
real_gap, gen_gap = real_regions["_gap"], gen_regions["_gap"]
for i, (gap, colour) in enumerate(((real_gap, ORANGE), (gen_gap, BLUE))):
    ax.bar(i, gap[0], width=0.5, color=colour,
           yerr=[[gap[0] - gap[1]], [gap[2] - gap[0]]], ecolor=INK_MUTED,
           capsize=4, error_kw={"linewidth": 1.2})
    ax.text(i, gap[2] + 0.25, f"{gap[0]:+.1f}", ha="center", fontsize=11, color=INK)
ax.set_xticks([0, 1], ["Real", "Generated"])
tidy(ax, "", "Framework agreement minus\nCDR agreement (points)")
ax.set_title("How strongly each separates\nscaffold from loops", pad=8)
ax.set_ylim(0, max(real_gap[2], gen_gap[2]) * 1.35)
overlap = real_gap[1] <= gen_gap[2] and gen_gap[1] <= real_gap[2]
print(f"  real gap {real_gap[0]:.1f} [{real_gap[1]:.1f},{real_gap[2]:.1f}]; "
      f"generated {gen_gap[0]:.1f} [{gen_gap[1]:.1f},{gen_gap[2]:.1f}]; "
      f"intervals overlap: {overlap}")

fig.suptitle("Framework against loop agreement, real and generated",
             fontsize=12.5, y=1.08)
finish(fig, "27_cdr_regions.png",
       f"Bars are means over 200 resampled sets of {n} chains; whiskers are 95% intervals. "
       "Real antibodies vary more sharply between scaffold and loops than the generated ones "
       "do — the model has the right shape, with the contrast flattened.")
print("CDR organisation figures done")

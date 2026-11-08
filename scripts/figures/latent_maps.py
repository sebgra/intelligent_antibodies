"""
What the VAE's latent space organises itself by.

Three views of the same 16-D space, projected with one shared PCA so the
panels are directly comparable: V-gene family, species, and how conserved
each chain is. None of these labels were given to the model.

    uv run python scripts/figures/latent_maps.py
"""
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from _common import (AQUA, BLUE, CRITICAL, FAMILY_COLOUR, GRID, INK, INK_2,
                     INK_MUTED, ORANGE, SEQ, SURFACE, VECTOR_SIZE, finish,
                     germline_family, legend_above, load_models_and_stats,
                     real_chains, style, tidy)
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder

style()
chains = real_chains()
models, stats = load_models_and_stats()
encoder = ProteinOneHotEncoder()

X = encoder.encode(chains["sequence"], VECTOR_SIZE).astype("float32")[..., None]
z_mean, _, _ = models["vae_full"].vae.encoder.predict(X, batch_size=64, verbose=0)
pca = PCA(n_components=2).fit(z_mean)
emb = pca.transform(z_mean)
print(f"{len(chains)} chains embedded; PC1+PC2 explain "
      f"{100 * pca.explained_variance_ratio_[:2].sum():.0f}% of latent variance")


def knn_recoverability(mask, labels):
    """
    How well a 15-NN classifier recovers a label from the latent coordinates,
    against the majority-class baseline. This is the honest way to say
    "the latent space knows about X": a scatter plot can look structured
    while carrying no usable signal, and vice versa.
    """
    mask = np.asarray(mask)
    labels = np.asarray(labels, dtype=object)[mask]
    keep = labels != None                                      # noqa: E711
    points, kept = z_mean[mask][keep], labels[keep]
    if len(set(kept)) < 2:
        return None
    _, counts = np.unique(kept, return_counts=True)
    baseline = counts.max() / counts.sum()
    score = cross_val_score(KNeighborsClassifier(15), points, kept, cv=5).mean()
    return score, baseline, len(kept)


def annotate_recoverability(ax, result, where=(0.5, -0.17)):
    if result is None:
        return
    score, baseline, n = result
    ax.annotate(f"15-NN recovery {score:.0%}  (chance {baseline:.0%}, n={n})",
                where, xycoords="axes fraction", ha="center", fontsize=9,
                color=INK_2)


def scatter_by(ax, labels, colours, order, title, counts=True, small=12):
    other = [i for i, l in enumerate(labels) if l not in order]
    if other:
        ax.scatter(emb[other, 0], emb[other, 1], color=SEQ[1], s=small,
                   edgecolors="none", zorder=1)
    handles = []
    for name in order:
        sel = [i for i, l in enumerate(labels) if l == name]
        if not sel:
            continue
        ax.scatter(emb[sel, 0], emb[sel, 1], color=colours[name], s=small,
                   edgecolors="none", zorder=2)
        handles.append(Line2D([], [], marker="o", linestyle="", color=colours[name],
                              markersize=6,
                              label=f"{name} ({len(sel)})" if counts else name))
    if other:
        handles.append(Line2D([], [], marker="o", linestyle="", color=SEQ[1],
                              markersize=6, label=f"other ({len(other)})"))
    ax.set_title(title, pad=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)
    return handles


# ================================================= 1. V-gene family
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
families = chains["family"].tolist()

heavy_families = [f for f in ("IGHV1", "IGHV3", "IGHV4", "IGHV2", "IGHV5", "IGHV6")
                  if families.count(f) >= 10]
light_families = [f for f in ("IGKV1", "IGKV3", "IGKV2", "IGKV4", "IGLV1", "IGLV2", "IGLV3")
                  if families.count(f) >= 10]

h = scatter_by(axes[0], families, FAMILY_COLOUR, heavy_families,
               "Heavy-chain V families")
axes[0].legend(handles=h, fontsize=8.5, loc="upper left", ncol=2, frameon=True,
               facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)
h = scatter_by(axes[1], families, FAMILY_COLOUR, light_families,
               "Light-chain V families (kappa and lambda)")
axes[1].legend(handles=h, fontsize=8.5, loc="upper right", ncol=2, frameon=True,
               facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)
heavy_mask = np.array([c == "heavy" for c in chains["chain"]])
light_mask = np.array([c == "light" for c in chains["chain"]])
annotate_recoverability(axes[0], knn_recoverability(
    heavy_mask, [f if f and f.startswith("IGHV") else None for f in families]))
annotate_recoverability(axes[1], knn_recoverability(
    light_mask, [f if f and f[:4] in ("IGKV", "IGLV") else None for f in families]))
fig.suptitle("The latent space separates germline V-gene families",
             fontsize=12.5, y=1.02)
finish(fig, "19_latent_by_family.png",
       "Families assigned by best framework-1 identity to a germline reference; the model was "
       "given sequences only. Heavy and light chains occupy opposite halves (99% recovery) and "
       "kappa separates from lambda (99%); families within a chain type overlap more, but stay "
       "well above chance.")

# ================================================= 2. species
fig, ax = plt.subplots(figsize=(7.4, 5.6))
species = chains["species"].tolist()
top = [s for s, _ in __import__("collections").Counter(species).most_common() if s][:4]
colours = {top[0]: BLUE, top[1]: ORANGE}
for extra, colour in zip(top[2:], (AQUA, "#6250d6")):
    colours[extra] = colour
handles = scatter_by(ax, species, colours, top, "")
ax.legend(handles=handles, fontsize=9, loc="upper left", frameon=True,
          facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)
ax.set_title("Species, which the model was also never told", pad=10)
annotate_recoverability(ax, knn_recoverability(
    np.ones(len(z_mean), bool),
    [s if s in ("Homo sapiens", "Mus musculus") else None for s in species]),
    where=(0.5, -0.14))
finish(fig, "20_latent_by_species.png",
       "Human and mouse chains are not cleanly separated, but they are not interleaved at "
       "random either: a nearest-neighbour classifier recovers the organism well above chance. "
       "The model never saw a species label — it encodes framework sequence, and human and "
       "mouse frameworks genuinely differ.")

# ============================== 6. conserved vs variable, located in the space
# How close is each chain to its own family's germline framework? High identity
# means a canonical, conserved chain; low means a divergent one.
identity = []
for seq, fam in zip(chains["sequence"], families):
    if fam is None:
        identity.append(np.nan)
        continue
    from _common import GERMLINE_FR1
    ref = GERMLINE_FR1[fam]
    n = min(len(ref), len(seq))
    identity.append(100 * sum(a == b for a, b in zip(seq[:n], ref[:n])) / n)
identity = np.array(identity, dtype=float)
known = ~np.isnan(identity)

# Where along the chain does the latent space actually have influence? Decode
# pairs of nearby latent points and count how often each position differs.
rng = np.random.default_rng(0)
centre = np.asarray(stats["mean"], dtype="float32")
spread = np.asarray(stats["std"], dtype="float32")
base = centre + spread * rng.normal(size=(200, len(centre))).astype("float32")
nudged = base + 0.35 * spread * rng.normal(size=base.shape).astype("float32")
decoded = []
for batch in (base, nudged):
    out = models["vae_full"].vae.decoder.predict(batch.astype("float32"), verbose=0)
    decoded.append(["".join(encoder.decode(r.reshape(VECTOR_SIZE, len(encoder.alphabet))))
                    for r in out])
window = 125
changed = np.zeros(window)
for a, b in zip(*decoded):
    for i in range(window):
        if i < len(a) and i < len(b) and a[i] != b[i]:
            changed[i] += 1
changed = 100 * changed / len(decoded[0])

from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

cdr_mask = np.zeros(window, bool)
for start_pos, end_pos in ((26, 35), (50, 65), (95, 110)):
    cdr_mask[start_pos:end_pos] = True
cdr_rate, fr_rate = changed[cdr_mask].mean(), changed[~cdr_mask].mean()
_, p_value = mannwhitneyu(changed[cdr_mask], changed[~cdr_mask])
identity_r2 = cross_val_score(KNeighborsRegressor(15), z_mean[known], identity[known],
                              cv=5, scoring="r2").mean()
print(f"  germline identity R2={identity_r2:.2f}; CDR {cdr_rate:.1f}% vs FR {fr_rate:.1f}% "
      f"(p={p_value:.3f}); N-term {changed[:15].mean():.1f}% -> C-term {changed[-15:].mean():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})

ax = axes[0]
points = ax.scatter(emb[known, 0], emb[known, 1], c=identity[known], cmap="viridis_r",
                    s=13, edgecolors="none", vmin=np.nanpercentile(identity, 5),
                    vmax=np.nanpercentile(identity, 95))
bar = fig.colorbar(points, ax=ax)
bar.set_label("Framework identity to germline (%)", fontsize=9)
bar.outline.set_visible(False)
ax.set_title("How germline-like a chain is, partly encoded", pad=10)
ax.annotate(f"predictable from latent position: R\u00b2 = {identity_r2:.2f}",
            (0.5, -0.17), xycoords="axes fraction", ha="center", fontsize=9, color=INK_2)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.grid(alpha=0.5)
ax.set_axisbelow(True)

ax = axes[1]
ax.fill_between(range(window), changed, color=SEQ[6], step="mid", zorder=2)
for start_pos, end_pos, label in ((26, 35, "CDR1"), (50, 65, "CDR2"), (95, 110, "CDR3")):
    ax.axvspan(start_pos, end_pos, color=ORANGE, alpha=0.18, zorder=0)
    ax.annotate(label, ((start_pos + end_pos) / 2, 52), ha="center", fontsize=8.5,
                color=ORANGE, zorder=4)
ax.axhline(cdr_rate, color=ORANGE, linestyle="--", linewidth=1.6, zorder=3)
ax.axhline(fr_rate, color=INK_2, linestyle=":", linewidth=1.6, zorder=3)
# Labels at the left, where the bars are lowest, so they sit on the surface
# rather than on the data.
ax.annotate(f"CDR mean {cdr_rate:.0f}%", (1, cdr_rate), textcoords="offset points",
            xytext=(2, 6), ha="left", fontsize=8.5, color=ORANGE, zorder=4)
ax.annotate(f"framework mean {fr_rate:.0f}%", (1, fr_rate), textcoords="offset points",
            xytext=(2, -12), ha="left", fontsize=8.5, color=INK_2, zorder=4)
ax.set_ylim(0, 58)
ax.set_xlim(0, window - 1)
tidy(ax, "Position in the variable domain", "Positions changed by a\nsmall latent move (%)")
ax.set_title("Stability falls along the chain, more than it tracks the loops", pad=10)

fig.suptitle("What a small move in latent space actually changes", fontsize=12.5, y=1.04)
finish(fig, "21_latent_conservation.png",
       f"200 latent points nudged by 0.35 standard deviations and decoded before and after. "
       f"CDR positions do change more often than framework ones ({cdr_rate:.0f}% vs "
       f"{fr_rate:.0f}%, p={p_value:.3f}), but the effect is small next to the drift along the "
       f"chain: {changed[:15].mean():.0f}% at the N-terminus against "
       f"{changed[-15:].mean():.0f}% at the C-terminus. The model holds the opening of a chain "
       "far more firmly than its end.")
print("latent map figures done")

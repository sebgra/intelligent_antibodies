"""
Shared pieces for the analysis figures.

These scripts inspect an already-trained model: nothing here retrains
anything, and every figure can be regenerated from the committed weights.
Generated candidates are cached on disk (see `sampled_candidates`) so that a
dozen figures do not each pay for their own sampling run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))

from intelligent_antibodies.modules.utils.chains import classify_chain  # noqa: E402
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder  # noqa: E402
from intelligent_antibodies.modules.utils.paths import (  # noqa: E402
    SABDAB_DIR, vae_latent_stats_path,
)

# Where the article's figures live. Override with IA_FIGURE_DIR.
OUT = Path(os.environ.get(
    "IA_FIGURE_DIR",
    "/home/sebastien/Bureau/sebgra.github.io/images/post_images/article_3",
))
CACHE = Path(os.environ.get("IA_FIGURE_CACHE", REPO / ".figure-cache"))

VECTOR_SIZE = 250

# ---------------------------------------------------------------- palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, VIOLET = "#eda100", "#e87ba4", "#6250d6"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CRITICAL = "#d03b3b"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "600",
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
        "lines.linewidth": 1.8, "pdf.fonttype": 42,
    })


def finish(fig, name: str, note: str | None = None, pdf: bool = False) -> None:
    if note:
        # Clear of the x-axis labels: at -0.01 the note printed on top of them.
        fig.text(0.5, -0.06, note, ha="center", fontsize=8.5, color=INK_MUTED, wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    if pdf:
        fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def tidy(ax, xlabel="", ylabel="", grid_axis="y"):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, alpha=0.6)
    ax.set_axisbelow(True)


def legend_above(ax, ncol=2, fontsize=9):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=ncol,
              fontsize=fontsize, frameon=False)


# ------------------------------------------------------------ chain anatomy
# Canonical framework-1 openings, one per V-gene family. Assignment is by best
# ungapped identity over the first ~25 residues: approximate next to a real
# germline aligner such as IMGT/V-QUEST, but it needs no database, and it
# recovers the expected repertoire usage on SAbDab (IGHV1/IGHV3/IGKV1 dominant,
# mean identity 0.84 over the 83% of chains it will commit to).
GERMLINE_FR1 = {
    "IGHV1": "QVQLVQSGAEVKKPGASVKVSCKAS", "IGHV2": "QVTLKESGPVLVKPTETLTLTCTVS",
    "IGHV3": "EVQLVESGGGLVQPGGSLRLSCAAS", "IGHV4": "QVQLQESGPGLVKPSETLSLTCTVS",
    "IGHV5": "EVQLVQSGAEVKKPGESLKISCKGS", "IGHV6": "QVQLQQSGPGLVKPSQTLSLTCAIS",
    "IGKV1": "DIQMTQSPSSLSASVGDRVTITCRAS", "IGKV2": "DIVMTQTPLSLPVTPGEPASISCRSS",
    "IGKV3": "EIVLTQSPGTLSLSPGERATLSCRAS", "IGKV4": "DIVMTQSPDSLAVSLGERATINCKSS",
    "IGLV1": "QSVLTQPPSVSGAPGQRVTISCTGS", "IGLV2": "QSALTQPASVSGSPGQSITISCTGT",
    "IGLV3": "SYELTQPPSVSVSPGQTASITCSGD",
}
FAMILY_COLOUR = {
    "IGHV1": BLUE, "IGHV3": SEQ[9], "IGHV4": SEQ[5], "IGHV2": SEQ[11],
    "IGHV5": SEQ[3], "IGHV6": SEQ[7],
    "IGKV1": ORANGE, "IGKV3": "#f08a5d", "IGKV2": "#c75227", "IGKV4": "#f5b08e",
    "IGLV1": AQUA, "IGLV2": "#168f63", "IGLV3": "#6fcfa8",
}


def germline_family(sequence: str, min_identity: float = 0.40):
    """Best-matching V-gene family, or None when nothing matches well."""
    best, best_id = None, 0.0
    for name, reference in GERMLINE_FR1.items():
        n = min(len(reference), len(sequence))
        if n < 15:
            continue
        identity = sum(a == b for a, b in zip(sequence[:n], reference[:n])) / n
        if identity > best_id:
            best, best_id = name, identity
    return best if best_id >= min_identity else None


def cysteine_anchors(sequence: str):
    """
    The two cysteines that form the variable domain's internal disulfide.

    In every immunoglobulin V domain one cysteine sits near the end of
    framework 1 (~position 22) and its partner in framework 3 (~position 88-92);
    the bond between them pins the two beta sheets together. Returned as
    ``(first, second)`` 0-based indices, or None where the pattern is absent.
    """
    positions = [i for i, residue in enumerate(sequence[:130]) if residue == "C"]
    if len(positions) < 2:
        return None
    first = next((p for p in positions if 15 <= p <= 35), None)
    if first is None:
        return None
    second = next((p for p in positions if first + 40 <= p <= first + 90), None)
    if second is None:
        return None
    return first, second


FR4_HEAVY = ("WGQG", "WGAG", "WGRG", "WGKG", "WGQT", "WGPG")
FR4_LIGHT = ("FGQG", "FGGG", "FGSG", "FGAG", "FGTG", "FGPG")


def variable_domain_end(sequence: str):
    """Index where framework 4 starts, i.e. where the variable domain closes."""
    hits = [sequence.find(m) for m in FR4_HEAVY + FR4_LIGHT]
    hits = [h for h in hits if 70 <= h <= 150]
    return min(hits) if hits else None


# ------------------------------------------------------------------- data
def real_chains() -> pd.DataFrame:
    """Antibody chains from SAbDab, with species, chain type and V family."""
    table = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
    table[["pdb", "chain_type_tag"]] = table["seq_id"].str.split("|", n=1, expand=True)
    table = table[table["chain_type_tag"] == "ab"].copy()
    # The species column mixes cases ("Homo sapiens" / "HOMO SAPIENS").
    table["species"] = (table["specie"].str.replace(r"\s*\(\d+\)", "", regex=True)
                        .str.strip().str.lower().str.capitalize())
    table["chain"] = [classify_chain(s) for s in table["sequence"]]
    table["family"] = [germline_family(s) for s in table["sequence"]]
    return table.reset_index(drop=True)


def load_models_and_stats(vector_size: int = VECTOR_SIZE):
    from intelligent_antibodies.modules.main_pipeline import load_models
    models = load_models(vector_size)
    stats = json.loads(vae_latent_stats_path(vector_size).read_text())
    return models, stats


def sampled_candidates(n: int = 400, chain_type: str | None = None,
                       temperature: float = 1.0, residue_temperature: float = 0.0,
                       vector_size: int = VECTOR_SIZE, seed: int = 0,
                       models=None, stats=None) -> list:
    """
    Generate (and cache) candidate sequences.

    Sampling is the slow part of every figure here and is identical between
    them, so results are memoised on disk keyed by their parameters.
    """
    from intelligent_antibodies.modules.utils.inference import generate_antibody_sequence

    key = f"gen-{vector_size}-{n}-{chain_type}-{temperature}-{residue_temperature}-{seed}"
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    if models is None or stats is None:
        models, stats = load_models_and_stats(vector_size)
    encoder = ProteinOneHotEncoder()
    np.random.seed(seed)
    out = []
    while len(out) < n:
        out += [s for s, _oh, _z, _c in generate_antibody_sequence(
            n=20, vae_model=models["vae_full"].vae, encoder=encoder,
            vector_size=vector_size, temperature=temperature, latent_stats=stats,
            residue_temperature=residue_temperature, chain_type=chain_type)]
    out = out[:n]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out))
    return out


# -------------------------------------------------------------- plausibility
def build_pssm(sequences, length: int = 130, alphabet: str = "ARNDCQEGHILKMFPSTWYV",
               pseudocount: float = 0.5) -> np.ndarray:
    """Per-position residue frequencies over a set of aligned-by-start chains."""
    index = {a: i for i, a in enumerate(alphabet)}
    counts = np.full((length, len(alphabet)), pseudocount)
    for seq in sequences:
        for position, residue in enumerate(seq[:length]):
            j = index.get(residue)
            if j is not None:
                counts[position, j] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def pssm_score(sequence: str, pssm: np.ndarray, background: np.ndarray,
               alphabet: str = "ARNDCQEGHILKMFPSTWYV") -> float:
    """
    Mean per-residue log-odds of a sequence under the PSSM.

    Positive means the sequence looks more like a real antibody chain than
    like a random draw from the background amino-acid composition. Scored
    only over the variable domain, where the alignment-by-start holds.
    """
    index = {a: i for i, a in enumerate(alphabet)}
    scores = []
    for position, residue in enumerate(sequence[:pssm.shape[0]]):
        j = index.get(residue)
        if j is None:
            continue
        scores.append(np.log2(pssm[position, j] / background[j]))
    return float(np.mean(scores)) if scores else float("nan")


def background_frequencies(sequences, alphabet: str = "ARNDCQEGHILKMFPSTWYV") -> np.ndarray:
    index = {a: i for i, a in enumerate(alphabet)}
    counts = np.full(len(alphabet), 0.5)
    for seq in sequences:
        for residue in seq:
            j = index.get(residue)
            if j is not None:
                counts[j] += 1
    return counts / counts.sum()

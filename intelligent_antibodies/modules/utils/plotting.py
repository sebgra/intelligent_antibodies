"""
Shared plot styling for the training curves.

Both training scripts used to draw with Matplotlib's defaults, which produced
figures that did not match anything else the project renders and, more
practically, put legends on top of the curves they were labelling. Everything
here is deliberately small: a validated colour palette, one `style()` call,
and a `legend_outside` helper so a legend can never cover data.

The palette is the same one the dashboard uses, chosen to stay distinguishable
under the common forms of colour-vision deficiency.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots, in fixed order -- never cycled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def style() -> None:
    """Apply the project's figure style to Matplotlib's global settings."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "600",
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "lines.linewidth": 1.8,
    })


def legend_outside(ax, ncol: int = 2, fontsize: float = 9) -> None:
    """
    Put the legend above the axes.

    Inside the axes it covered the curves -- on a validation curve that swings
    between 0.1 and 0.7 there is no safe corner to hide it in.
    """
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=ncol,
              fontsize=fontsize, frameon=False)


def tidy(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)


def save_history(history, path: Path) -> None:
    """
    Write the raw per-epoch metrics next to the figure.

    Without this, re-drawing a training curve means retraining the model: the
    numbers exist only inside the PNG. A few kilobytes of JSON makes the
    figures reproducible on their own.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {k: [float(v) for v in values] for k, values in history.history.items()},
        indent=2,
    ))

"""
Centralized, cwd-independent paths for the project.

Every path used to be hardcoded relative to wherever a script happened to be
launched from (``"../data/SAbDab/sequences.csv"``, ``"data/SAbDab/..."``,
etc.), which only worked from one specific working directory per script.
Everything below is anchored to this repository's root instead, resolved
from this file's own location, so the same code works no matter where it's
invoked from (a notebook, `uv run`, the Streamlit app, or a cron job).
"""
from pathlib import Path

# intelligent_antibodies/modules/utils/paths.py -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
SABDAB_DIR = DATA_DIR / "SAbDab"
COVABDAB_DIR = DATA_DIR / "CoV-AbDab"
RUN_DIR = REPO_ROOT / "run"
MODELS_DIR = RUN_DIR / "models"
VAE_MODEL_DIR = MODELS_DIR / "vae"
SIAMESE_MODEL_DIR = MODELS_DIR / "siamese"
PLOTS_DIR = REPO_ROOT / "plots"


def vae_weights_path(vector_size: int = 200) -> Path:
    """Base path VAE.save()/VAE.reload() split into -encoder.keras/-decoder.keras."""
    return VAE_MODEL_DIR / f"vae-one-hot-{vector_size}"


def siamese_weights_path(vector_size: int = 200) -> Path:
    return SIAMESE_MODEL_DIR / f"one-hot-{vector_size}-model.h5"


def trained_models_available(vector_size: int = 200) -> bool:
    """True if both the VAE and Siamese weight files exist on disk."""
    vae_base = vae_weights_path(vector_size)
    return (
        vae_base.with_name(vae_base.name + "-encoder.keras").exists()
        and vae_base.with_name(vae_base.name + "-decoder.keras").exists()
        and siamese_weights_path(vector_size).exists()
    )


def available_vector_sizes() -> list[int]:
    """
    Every vector size with a complete set of trained weights on disk.

    Weight files are named per vector size (see the two path helpers above),
    so the sizes that can actually be loaded are whatever has been trained.
    Lets a caller -- e.g. the dashboard's sequence-length control -- say which
    sizes are usable instead of silently falling back to simulated results.
    """
    sizes = {
        int(stem) for p in VAE_MODEL_DIR.glob("vae-one-hot-*-encoder.keras")
        if (stem := p.name[len("vae-one-hot-"):-len("-encoder.keras")]).isdigit()
    }
    return sorted(s for s in sizes if trained_models_available(s))

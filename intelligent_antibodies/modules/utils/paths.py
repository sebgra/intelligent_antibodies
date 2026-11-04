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


def siamese_calibration_path(vector_size: int = 200) -> Path:
    """
    Where `scripts/train_siamese.py` records what the trained classifier's
    scores actually look like (held-out AUC, score percentiles, useful
    operating points).

    Without this the dashboard's "interaction score threshold" is a guess: a
    classifier whose scores never leave 0.47-0.69 can never clear 0.80, so a
    run returns nothing with no hint as to why. The app reads this file to
    show the model's real score range and recommend a threshold.
    """
    return SIAMESE_MODEL_DIR / f"one-hot-{vector_size}-calibration.json"


def vae_latent_stats_path(vector_size: int = 200) -> Path:
    """
    Where `scripts/train_vae.py` records the VAE's aggregate posterior
    (per-dimension mean/std of `z_mean` over the training set).

    Sampling assumes the latent space matches the N(0, 1) prior. When the
    reconstruction term dominates the KL term that is not true, and sampling
    from N(0, 1) lands in regions the decoder never saw. Generation uses these
    statistics to sample where the data actually lives instead.
    """
    return VAE_MODEL_DIR / f"vae-one-hot-{vector_size}-latent-stats.json"


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

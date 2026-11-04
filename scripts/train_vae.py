"""
Train the antibody-sequence VAE (the generator half of the pipeline).

Loads one-hot encoded antibody sequences (data/SAbDab, antibody chains
only), trains a convolutional VAE (see
`intelligent_antibodies/modules/models/VAEFull.py`), and saves the trained
encoder/decoder under run/models/vae/.

Alongside the weights it writes `vae-one-hot-<size>-latent-stats.json`: the
per-dimension mean/std of the encoder's output over the training set.
Generation samples the latent space, and that only produces plausible
sequences if it samples where the data actually lies -- see
`inference.sample_latent`.

Compute is capped at 50% of the machine by default (GPU memory and CPU
threads) -- see `intelligent_antibodies/modules/utils/gpu.py`, or set
`IA_COMPUTE_FRACTION`.

Usage
-----
    uv run python scripts/train_vae.py                       # defaults
    uv run python scripts/train_vae.py --epochs 10 --limit 200  # quick smoke test

See README.md, "Training the models", for expected run time and how the
dashboard picks this up automatically once trained.
"""
import argparse
import json
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from intelligent_antibodies.modules.dataset import AntibodyOneHotProtDataset
from intelligent_antibodies.modules.models.VAEFull import VAEFull
from intelligent_antibodies.modules.utils.gpu import configure_compute
from intelligent_antibodies.modules.utils.plotting import (AQUA, BLUE, ORANGE,
                                                           legend_outside, save_history,
                                                           style, tidy)
from intelligent_antibodies.modules.utils.paths import (
    PLOTS_DIR,
    VAE_MODEL_DIR,
    vae_latent_stats_path,
    vae_weights_path,
)


class KLWarmup(keras.callbacks.Callback):
    """
    Ramp the KL weight from 0 to 1 over the first `warmup_epochs` epochs.

    A full KL penalty from step 0 collapses the latent space before the
    decoder can learn to use it (every dimension matches the prior and
    carries no information); no penalty at all leaves the aggregate posterior
    nowhere near the prior the sampler draws from. Warming up gets both.
    """

    def __init__(self, warmup_epochs: int):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0:
            weight = 1.0
        else:
            weight = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.model.kl_weight.assign(weight)


class EpochStats(keras.callbacks.Callback):
    """Per-epoch progress bar plus the stats worth watching on a VAE."""

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        done = epoch + 1
        filled = int(24 * done / max(self.total_epochs, 1))
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = time.time() - self.t0
        eta = (elapsed / done) * (self.total_epochs - done)
        # `residue_accuracy` is the one to watch: the fraction of positions
        # the decoder reconstructs exactly. Loss alone says little, since the
        # KL weight is still moving during warmup.
        # Leading newline: Keras' progress bar ends its line with a carriage
        # return, so without this the stats land in the middle of that bar.
        print(
            f"\n  [{bar}] epoch {done}/{self.total_epochs}  "
            f"loss {logs.get('loss', float('nan')):.2f}  "
            f"recon {logs.get('reconstruction_loss', float('nan')):.2f}  "
            f"kl {logs.get('kl_loss', float('nan')):.3f}  "
            f"residue_acc {logs.get('residue_accuracy', 0.0):.4f}  "
            f"val_residue_acc {logs.get('val_residue_accuracy', 0.0):.4f}  "
            f"kl_w {float(keras.ops.convert_to_numpy(self.model.kl_weight)):.2f}  "
            f"[{elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m left]",
            flush=True,
        )


def plot_history(history, out_path) -> None:
    """Training curves, styled to match the rest of the project's figures."""
    style()
    h = history.history

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(h["loss"], color=BLUE, label="Total loss")
    ax.plot(h["reconstruction_loss"], color=ORANGE, label="Reconstruction")
    ax.plot(h["kl_loss"], color=AQUA, label="KL divergence")
    tidy(ax, "Epoch", "VAE loss")
    ax.set_ylim(0, max(h["loss"]) * 1.06)
    legend_outside(ax, ncol=3)

    ax = axes[1]
    if "residue_accuracy" in h:
        ax.plot(h["residue_accuracy"], color=BLUE, label="Train")
        if "val_residue_accuracy" in h:
            ax.plot(h["val_residue_accuracy"], color=ORANGE, label="Validation")
        tidy(ax, "Epoch", "Exact-residue reconstruction")
        ax.set_ylim(0, 1.06)
        legend_outside(ax)

    fig.suptitle("VAE training history", fontsize=12.5, y=1.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    # 16, not 2: a 2-D latent space can only trace a 2-D manifold of
    # sequences, so repeated sampling returned near-copies of a few
    # frameworks -- too few DISTINCT candidates to ever fill a request.
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0,
                         help="Weight on the KL term (higher = latent space closer to the prior).")
    parser.add_argument("--free-bits", type=float, default=0.05,
                         help="Per-dimension KL floor in nats, to stop dimensions collapsing.")
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # There are only ~1,500 antibody chains, so validation reconstruction
    # plateaus (around 0.87 exact-residue accuracy) by roughly epoch 40 while
    # training accuracy keeps climbing past 0.98. Stopping on the held-out
    # measure keeps the decoder that generalizes rather than the one that has
    # memorized the training chains.
    parser.add_argument("--patience", type=int, default=25,
                         help="Early-stopping patience on validation loss (0 disables).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use only the first N antibody sequences (for a quick smoke test).")
    parser.add_argument("--compute-fraction", type=float, default=None,
                         help="Fraction of GPU memory / CPU cores to use (default 0.5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_compute(args.compute_fraction)
    keras.utils.set_random_seed(args.seed)

    print(f"Loading antibody sequences (vector_size={args.vector_size})...")
    x_data, _y_data, vector_size, alphabet_size = AntibodyOneHotProtDataset.get_data(args.vector_size)
    if args.limit:
        x_data = x_data[: args.limit]
    print(f"{len(x_data)} antibody sequence(s), alphabet_size={alphabet_size}.")

    # Train/validation split is now actually USED as one. It used to be split
    # and then immediately concatenated back together, so every epoch trained
    # on 100% of the data and there was no held-out signal at all.
    x_train, x_val = train_test_split(x_data, test_size=args.test_size, random_state=args.seed)
    x_train = x_train.reshape(*x_train.shape, 1).astype("float32")
    x_val = x_val.reshape(*x_val.shape, 1).astype("float32")

    print(f"Building VAE (Conv2D encoder / Conv2DTranspose decoder, latent dim {args.latent_dim})...")
    vae_full = VAEFull(vector_size, alphabet_size, latent_dim=args.latent_dim)
    vae_full.vae.beta = args.beta
    vae_full.vae.free_bits = args.free_bits
    vae_full.vae.kl_warmup_epochs = args.kl_warmup_epochs
    # clipnorm, because the KL term can spike hard in the first few steps at a
    # long sequence window; without it the run diverged to NaN at 400 aa.
    vae_full.vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    )
    print(f"  encoder {vae_full.encoder.count_params():,} params, "
          f"decoder {vae_full.decoder.count_params():,} params; "
          f"{len(x_train):,} train / {len(x_val):,} validation sequences.")

    callbacks = [KLWarmup(args.kl_warmup_epochs), EpochStats(args.epochs)]
    if args.patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=args.patience,
                restore_best_weights=True,
            )
        )

    print(f"Training for up to {args.epochs} epoch(s), batch size {args.batch_size}...")
    history = vae_full.vae.fit(
        x_train, epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(x_val,),
        callbacks=callbacks,
        verbose=1,
    )

    out_path = vae_weights_path(vector_size)
    VAE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vae_full.vae.save(str(out_path))
    print(f"Saved encoder/decoder to {out_path}-encoder.keras / -decoder.keras")

    # Where the data actually lives in latent space, for the sampler.
    stats = vae_full.vae.latent_statistics(x_train, batch_size=args.batch_size)
    stats["beta"] = args.beta
    stats["vector_size"] = int(vector_size)
    stats_path = vae_latent_stats_path(vector_size)
    stats_path.write_text(json.dumps(stats, indent=2))
    spread = np.asarray(stats["std"])
    print(f"Latent statistics written to {stats_path}")
    print(f"  aggregate posterior std per dim: min {spread.min():.3f}, "
          f"mean {spread.mean():.3f}, max {spread.max():.3f} "
          f"(a std far from 1.0 is why sampling N(0,1) alone decodes poorly)")
    print(f"  final residue reconstruction accuracy: "
          f"{history.history.get('val_residue_accuracy', [float('nan')])[-1]:.4f} (validation)")

    plot_path = PLOTS_DIR / "VAE_training_curve.png"
    plot_history(history, plot_path)
    save_history(history, PLOTS_DIR / "VAE_training_history.json")
    print(f"Saved training curve to {plot_path}")


if __name__ == "__main__":
    main()

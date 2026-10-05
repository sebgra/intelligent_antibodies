"""
Train the antibody-sequence VAE (the generator half of the pipeline).

Loads one-hot encoded antibody sequences (data/SAbDab, antibody chains
only), trains a convolutional VAE with a 2-D latent space (see
`intelligent_antibodies/modules/models/VAEFull.py`), and saves the trained
encoder/decoder under run/models/vae/.

Usage
-----
    uv run python scripts/train_vae.py                       # defaults
    uv run python scripts/train_vae.py --epochs 10 --limit 200  # quick smoke test

See README.md, "Training the models", for expected run time and how the
dashboard picks this up automatically once trained.
"""
import argparse

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from intelligent_antibodies.modules.dataset import AntibodyOneHotProtDataset
from intelligent_antibodies.modules.models.VAE import VAE
from intelligent_antibodies.modules.models.VAEFull import VAEFull
from intelligent_antibodies.modules.utils.paths import PLOTS_DIR, VAE_MODEL_DIR, vae_weights_path


def _configure_gpu_memory_growth() -> None:
    """Avoid TensorFlow grabbing all GPU memory up front, and let it fail
    gracefully (falling back to whatever's available) rather than crash."""
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not set memory growth on {gpu}: {exc}")


def plot_history(history, out_path) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["reconstruction_loss"], label="Reconstruction loss")
    plt.plot(history.history["kl_loss"], label="KL loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("VAE loss", fontsize=14)
    plt.title("VAE training history", fontsize=17)
    plt.legend(fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None,
                         help="Use only the first N antibody sequences (for a quick smoke test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_gpu_memory_growth()

    print(f"Loading antibody sequences (vector_size={args.vector_size})...")
    x_data, _y_data, vector_size, alphabet_size = AntibodyOneHotProtDataset.get_data(args.vector_size)
    if args.limit:
        x_data = x_data[: args.limit]
    print(f"{len(x_data)} antibody sequence(s), alphabet_size={alphabet_size}.")

    x_train, x_test = train_test_split(x_data, test_size=args.test_size, random_state=args.seed)
    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)
    dataset = np.concatenate([x_train, x_test], axis=0).astype("float32")

    print("Building VAE (Conv2D encoder / Conv2DTranspose decoder, latent dim 2)...")
    vae_full = VAEFull(vector_size, alphabet_size)
    vae_full.vae.compile(optimizer=keras.optimizers.Adam())

    print(f"Training for {args.epochs} epoch(s), batch size {args.batch_size}...")
    history = vae_full.vae.fit(dataset, epochs=args.epochs, batch_size=args.batch_size)

    out_path = vae_weights_path(vector_size)
    VAE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vae_full.vae.save(str(out_path))
    print(f"Saved encoder/decoder to {out_path}-encoder.keras / -decoder.keras")

    plot_path = PLOTS_DIR / "VAE_training_curve.png"
    plot_history(history, plot_path)
    print(f"Saved training curve to {plot_path}")


if __name__ == "__main__":
    main()

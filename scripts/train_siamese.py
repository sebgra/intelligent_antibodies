"""
Train the Siamese antibody-antigen interaction classifier (the
discriminator half of the pipeline).

Loads the equilibrated (class-balanced) antibody/antigen pair dataset (see
`intelligent_antibodies/modules/dataset.py::EquilibratedDataset`), trains the
Conv1D+BiGRU Siamese network (see
`intelligent_antibodies/modules/models/SiameseInteractionClassifier.py`), and
saves the trained model under run/models/siamese/.

Usage
-----
    uv run python scripts/train_siamese.py                        # defaults
    uv run python scripts/train_siamese.py --epochs 5 --limit 500   # quick smoke test

See README.md, "Training the models", for expected run time and how the
dashboard picks this up automatically once trained.
"""
import argparse

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

from intelligent_antibodies.modules.dataset import EquilibratedDataset
from intelligent_antibodies.modules.models.SiameseInteractionClassifier import SiameseInteractionClassifier
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.paths import PLOTS_DIR, SIAMESE_MODEL_DIR, siamese_weights_path


def _configure_gpu_memory_growth() -> None:
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not set memory growth on {gpu}: {exc}")


def plot_history(history, curve_path, metrics_path) -> None:
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(history.history["accuracy"], label="Accuracy train")
    plt.plot(history.history["val_accuracy"], label="Accuracy validation")
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=14)

    plt.subplot(122)
    plt.plot(history.history["loss"], label="Loss train")
    plt.plot(history.history["val_loss"], label="Loss validation")
    plt.ylabel("Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=14)
    plt.suptitle("Siamese training", fontsize=17)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(curve_path)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(history.history["f1"], label="F1-score train")
    plt.plot(history.history["val_f1"], label="F1-score validation")
    plt.ylabel("F1-score", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=14)

    plt.subplot(122)
    plt.plot(history.history["mcc"], label="MCC train")
    plt.plot(history.history["val_mcc"], label="MCC validation")
    plt.ylabel("MCC", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=14)
    plt.suptitle("Siamese training", fontsize=17)
    plt.savefig(metrics_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--filters", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5,
                         help="Early-stopping patience on validation loss.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use only the first N training pairs (for a quick smoke test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_gpu_memory_growth()

    print(f"Loading equilibrated antibody/antigen pairs (vector_size={args.vector_size})...")
    encoder = ProteinOneHotEncoder()
    X, y, vector_size, alphabet_size = EquilibratedDataset(encoder).getdata(args.vector_size)
    x_ab, x_ag = X
    if args.limit:
        x_ab, x_ag, y = x_ab[: args.limit], x_ag[: args.limit], y[: args.limit]
    print(f"{len(y)} pair(s), alphabet_size={alphabet_size}, "
          f"positive rate={float(y.mean()):.2f} (should be ~0.5, equilibrated).")

    x_train_ab, x_test_ab, x_train_ag, x_test_ag, y_train, y_test = train_test_split(
        x_ab, x_ag, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )

    input_dimensions = (vector_size, alphabet_size)
    seq_input1 = layers.Input(shape=input_dimensions, name="seq_ag")
    seq_input2 = layers.Input(shape=input_dimensions, name="seq_ab")
    siamese = SiameseInteractionClassifier(args.filters, seq_input1, seq_input2)

    SIAMESE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = siamese_weights_path(vector_size)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        str(out_path), monitor="val_mcc", mode="max", save_best_only=True,
    )
    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience)

    print(f"Training for up to {args.epochs} epoch(s), batch size {args.batch_size}...")
    history = siamese.model.fit(
        [x_train_ab, x_train_ag], y_train, epochs=args.epochs,
        callbacks=[checkpoint_callback, earlystop_callback],
        batch_size=args.batch_size, verbose=1, validation_split=args.validation_split,
    )

    test_metrics = siamese.model.evaluate([x_test_ab, x_test_ag], y_test, verbose=0)
    print(f"Test metrics (loss, accuracy, f1, mcc): {test_metrics}")
    print(f"Best model (by val_mcc) saved to {out_path}")

    plot_history(
        history, PLOTS_DIR / "Siamese_training_curve.png", PLOTS_DIR / "Siamese_training_metrics.png",
    )
    print(f"Saved training curves to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

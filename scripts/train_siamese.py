"""
Train the Siamese antibody-antigen interaction classifier (the
discriminator half of the pipeline).

Loads the equilibrated (class-balanced) antibody/antigen pair dataset (see
`intelligent_antibodies/modules/dataset.py::EquilibratedDataset`), trains the
Conv1D+BiGRU Siamese network (see
`intelligent_antibodies/modules/models/SiameseInteractionClassifier.py`), and
saves the trained model under run/models/siamese/.

It also writes a **calibration file** next to the weights
(`one-hot-<size>-calibration.json`): held-out AUC, the score percentiles the
model actually produces, and useful operating points. The dashboard reads it
so its "interaction score threshold" slider reflects what this model can
really output -- the previous model's scores never left 0.47-0.69, which made
the default 0.80 threshold unreachable and every run come back empty.

Compute is capped at 50% of the machine by default (GPU memory and CPU
threads) -- see `intelligent_antibodies/modules/utils/gpu.py`, or set
`IA_COMPUTE_FRACTION`.

Usage
-----
    uv run python scripts/train_siamese.py                        # defaults
    uv run python scripts/train_siamese.py --epochs 5 --limit 500   # quick smoke test

See README.md, "Training the models", for expected run time and how the
dashboard picks this up automatically once trained.
"""
import argparse
import json
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from intelligent_antibodies.modules.dataset import EquilibratedDataset
from intelligent_antibodies.modules.models.SiameseInteractionClassifier import (
    SiameseInteractionClassifier,
)
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.gpu import configure_compute
from intelligent_antibodies.modules.utils.plotting import (BLUE, INK_MUTED, ORANGE,
                                                           legend_outside, save_history,
                                                           style, tidy)
from intelligent_antibodies.modules.utils.paths import (
    PLOTS_DIR,
    SIAMESE_MODEL_DIR,
    siamese_calibration_path,
    siamese_weights_path,
)


class EpochStats(keras.callbacks.Callback):
    """
    A one-line-per-epoch progress report with the stats that actually matter
    here, on top of Keras' own per-batch progress bar.

    Accuracy alone hid the previous model's failure: it sat at chance on a
    balanced set whether the model had learned anything or not. AUC is
    threshold-free, so a collapsed model reads as ~0.50 immediately, and the
    arrow marks epochs that actually improved held-out AUC.
    """

    def __init__(self, total_epochs: int, monitor: str = "val_mcc"):
        super().__init__()
        self.total_epochs = total_epochs
        self.monitor = monitor
        self.monitor_label = monitor.replace("val_", "")
        self.best = -1.0
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_auc = float(logs.get("val_auc", 0.0))
        current = float(logs.get(self.monitor, 0.0))
        improved = current > self.best
        self.best = max(self.best, current)
        done = epoch + 1
        # A plain text progress bar, so a long run is readable in a log file
        # or a scrollback buffer without a live terminal.
        filled = int(24 * done / max(self.total_epochs, 1))
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = time.time() - self.t0
        eta = (elapsed / done) * (self.total_epochs - done)
        # Leading newline: Keras' progress bar ends its line with a carriage
        # return, so without this the stats land in the middle of that bar.
        print(
            f"\n  [{bar}] epoch {done}/{self.total_epochs}  "
            f"loss {logs.get('loss', float('nan')):.4f}  val_loss {logs.get('val_loss', float('nan')):.4f}  "
            f"val_auc {val_auc:.4f}  val_mcc {logs.get('val_mcc', 0.0):+.4f}  "
            f"val_f1 {logs.get('val_f1', 0.0):.4f}  "
            f"best_{self.monitor_label} {self.best:.4f}{'  <- improved' if improved else ''}  "
            f"[{elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m left]",
            flush=True,
        )


def tuned_threshold(y_true, scores, metric="mcc"):
    """
    The decision threshold that maximises MCC (or F1) on these predictions.

    0.5 is only the right cut when the classes are balanced AND the model is
    calibrated; neither holds here. Choosing the cut on the VALIDATION set and
    reporting it on the test set keeps the reported number honest.
    """
    from sklearn.metrics import f1_score, matthews_corrcoef

    score_fn = matthews_corrcoef if metric == "mcc" else f1_score
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_value, best_cut = -2.0, 0.5
    for cut in grid:
        predicted = (scores >= cut).astype(int)
        if predicted.sum() == 0 or predicted.sum() == len(predicted):
            continue
        value = score_fn(y_true, predicted)
        if value > best_value:
            best_value, best_cut = float(value), float(cut)
    return best_cut, best_value


def write_calibration(model, x_test_ab, x_test_ag, y_test, out_path, vector_size, batch_size,
                      validation_data=None):
    """
    Record what this model's scores look like on held-out pairs.

    The dashboard needs this to turn its threshold slider into something
    meaningful: a threshold above the model's observed maximum can never be
    met, and "recommended" below is the score that maximizes Youden's J (the
    best true-positive/false-positive trade-off on this test set).
    """
    from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score, roc_curve)

    scores = model.predict([x_test_ab, x_test_ag], batch_size=batch_size, verbose=0).ravel()
    auc = float(roc_auc_score(y_test, scores))
    ap = float(average_precision_score(y_test, scores))

    # Pick the decision cut on validation, then report what it achieves on
    # test. Tuning and reporting on the same split would flatter the result.
    if validation_data is not None:
        (val_ab, val_ag), y_val = validation_data
        val_scores = model.predict([val_ab, val_ag], batch_size=batch_size, verbose=0).ravel()
        mcc_cut, _ = tuned_threshold(y_val, val_scores, "mcc")
        f1_cut, _ = tuned_threshold(y_val, val_scores, "f1")
    else:
        mcc_cut, f1_cut = 0.5, 0.5

    def at(cut):
        pred = (scores >= cut).astype(int)
        return {
            "threshold": round(float(cut), 4),
            "mcc": round(float(matthews_corrcoef(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        }

    decision = {"at_0.5": at(0.5), "tuned_for_mcc": at(mcc_cut), "tuned_for_f1": at(f1_cut)}

    fpr, tpr, thresholds = roc_curve(y_test, scores)
    youden = int(np.argmax(tpr - fpr))
    recommended = float(thresholds[youden])

    # Precision at a few candidate cut-offs: for generation, precision is what
    # matters -- a kept candidate should actually be an interactor.
    operating_points = []
    for cut in np.quantile(scores, [0.5, 0.75, 0.9, 0.95, 0.99]):
        kept = scores >= cut
        if kept.sum() == 0:
            continue
        operating_points.append({
            "threshold": round(float(cut), 4),
            "kept_fraction": round(float(kept.mean()), 4),
            "precision": round(float(y_test[kept].mean()), 4),
        })

    # For GENERATION, precision is what matters: a kept candidate should
    # actually be an interactor, and sampling can always draw more. Youden's J
    # above balances sensitivity against specificity, which is the right
    # answer for classifying a fixed set of pairs and too permissive for
    # picking candidates out of an unlimited pool.
    #
    # The floor is 25% of pairs kept, not 10%. Squeezing to the top decile
    # bought almost nothing (precision 0.755 vs 0.736) while pushing the cut
    # from 0.95 to 0.996 -- and generated candidates score lower than
    # held-out real positives, so that stricter cut turned a run that finds
    # 20 candidates in 11 batches into one that finds 5 in 80.
    selection = max(
        (op for op in operating_points if op["kept_fraction"] >= 0.25),
        key=lambda op: op["precision"],
        default=None,
    )

    calibration = {
        "vector_size": int(vector_size),
        "n_test_pairs": int(len(y_test)),
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "score_min": round(float(scores.min()), 4),
        "score_max": round(float(scores.max()), 4),
        "score_mean": round(float(scores.mean()), 4),
        "score_std": round(float(scores.std()), 4),
        "percentiles": {
            str(int(q * 100)): round(float(np.quantile(scores, q)), 4)
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "decision": decision,
        "mcc_threshold": round(float(mcc_cut), 4),
        "f1_threshold": round(float(f1_cut), 4),
        "recommended_threshold": round(recommended, 4),
        "selection_threshold": selection["threshold"] if selection else round(recommended, 4),
        "selection_precision": selection["precision"] if selection else None,
        "recommended_precision": round(
            float(y_test[scores >= recommended].mean()) if (scores >= recommended).any() else 0.0, 4
        ),
        "positive_score_mean": round(float(scores[y_test == 1].mean()), 4),
        "negative_score_mean": round(float(scores[y_test == 0].mean()), 4),
        "operating_points": operating_points,
    }
    out_path.write_text(json.dumps(calibration, indent=2))
    return calibration


def plot_history(history, curve_path, metrics_path) -> None:
    """Training curves, styled to match the rest of the project's figures."""
    style()
    h = history.history

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (key, label) in zip(axes, (("accuracy", "Accuracy"), ("loss", "Loss"))):
        ax.plot(h[key], color=BLUE, label=f"{label} (train)")
        ax.plot(h[f"val_{key}"], color=ORANGE, label=f"{label} (validation)")
        tidy(ax, "Epoch", label)
        legend_outside(ax)
    fig.suptitle("Interaction classifier — loss and accuracy", fontsize=12.5, y=1.06)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(curve_path, bbox_inches="tight")
    plt.close(fig)

    panels = [("f1", "F1-score", None), ("mcc", "MCC", None), ("auc", "ROC AUC", 0.5)]
    panels = [p for p in panels if p[0] in h]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, label, floor) in zip(axes, panels):
        ax.plot(h[key], color=BLUE, label="Train")
        ax.plot(h[f"val_{key}"], color=ORANGE, label="Validation")
        if floor is not None:
            ax.axhline(floor, color=INK_MUTED, linestyle=":", linewidth=1.4)
            ax.annotate("chance", (0.99, floor), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 4), ha="right",
                        fontsize=8.5, color=INK_MUTED)
        tidy(ax, "Epoch", label)
        # Headroom so the legend above the axes never meets the curves.
        ax.set_ylim(min(0.0, min(h[f"val_{key}"])) - 0.05, 1.08)
        legend_outside(ax)
    fig.suptitle("Interaction classifier — held-out metrics", fontsize=12.5, y=1.06)
    fig.tight_layout()
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 400, not 200: median antibody/antigen length is ~218/220 residues, so 200
    # truncated over half of every sequence away. Note the VAE must be trained at
    # the SAME size (scripts/train_vae.py --vector-size) or the pair won't be
    # discoverable -- weights are stored per length and the dashboard only offers
    # sizes that have both (see utils/paths.py::available_vector_sizes).
    parser.add_argument("--vector-size", type=int, default=400)
    parser.add_argument("--filters", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-split", type=float, default=0.15)
    # 1e-3, not 1e-4: at 1e-4 this network never escaped the mean predictor
    # before early stopping, which is what pinned every score near 0.55.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed-dim", type=int, default=32)
    # 3 negatives per positive: there are 805,803 available negatives against
    # 3,989 positives, and a 1:1 balance discarded all but 3,989 of them.
    parser.add_argument("--negative-ratio", type=float, default=3.0,
                         help="Negatives kept per positive pair (1.0 = the old strict balance).")
    parser.add_argument("--split-by", choices=("antibody", "random"), default="antibody",
                         help="'antibody' holds out whole antibodies (honest estimate of how the "
                              "model will score NEWLY generated sequences); 'random' splits pairs.")
    parser.add_argument("--patience", type=int, default=12,
                         help="Early-stopping patience on the monitored metric.")
    parser.add_argument("--monitor", default="val_mcc",
                         choices=("val_mcc", "val_auc", "val_f1"),
                         help="Metric that checkpointing and early stopping follow. MCC is the "
                              "balanced summary of the confusion matrix and is now computed "
                              "over the whole epoch, so it is safe to steer on.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Use a random balanced subsample of N pairs (for a quick smoke test).")
    parser.add_argument("--compute-fraction", type=float, default=None,
                         help="Fraction of GPU memory / CPU cores to use (default 0.5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_compute(args.compute_fraction)
    keras.utils.set_random_seed(args.seed)

    print(f"Loading equilibrated antibody/antigen pairs (vector_size={args.vector_size}, "
          f"{args.negative_ratio:g} negative(s) per positive)...")
    encoder = ProteinOneHotEncoder()
    X, y, vector_size, alphabet_size, groups = EquilibratedDataset(encoder).getdata(
        args.vector_size, negative_ratio=args.negative_ratio, return_groups=True,
        random_state=args.seed,
    )
    x_ab, x_ag = X
    if args.limit:
        # EquilibratedDataset concatenates all positives then all negatives, so
        # slicing the head returned a single-class subset -- and the stratified
        # split below then died on it. Shuffle first so --limit stays balanced.
        order = np.random.default_rng(args.seed).permutation(len(y))[: args.limit]
        x_ab, x_ag, y, groups = x_ab[order], x_ag[order], y[order], groups[order]
    print(f"{len(y)} pair(s), alphabet_size={alphabet_size}, "
          f"positive rate={float(y.mean()):.2f}.")

    if args.split_by == "antibody":
        # Hold out whole antibodies, not random pairs: the same antibody
        # appears in many pairs, so a random split lets the model recognize a
        # memorized antibody at test time and reports an AUC that generation
        # (which invents unseen antibodies) will never live up to.
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        train_idx, test_idx = next(splitter.split(x_ab, y, groups=groups))
        print(f"Grouped split by antibody: {len(set(groups[train_idx]))} train / "
              f"{len(set(groups[test_idx]))} test antibodies, no overlap.")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size, random_state=args.seed, stratify=y,
        )

    # Carve the validation set out explicitly instead of letting
    # `fit(validation_split=...)` do it. Keras takes the LAST fraction of the
    # arrays WITHOUT shuffling, and the dataset arrives as all positives
    # followed by all negatives -- so the validation set came out entirely
    # negative, every AUC was reported as 0.0000, and both early stopping and
    # the best-weights checkpoint were steering on a meaningless number.
    # Splitting by group here also keeps validation antibodies out of
    # training, which `validation_split` could never do.
    if args.validation_split > 0:
        if args.split_by == "antibody":
            inner = GroupShuffleSplit(
                n_splits=1, test_size=args.validation_split, random_state=args.seed
            )
            fit_idx, val_idx = next(
                inner.split(train_idx, y[train_idx], groups=groups[train_idx])
            )
        else:
            fit_idx, val_idx = train_test_split(
                np.arange(len(train_idx)), test_size=args.validation_split,
                random_state=args.seed, stratify=y[train_idx],
            )
        val_idx_abs, train_idx = train_idx[val_idx], train_idx[fit_idx]
        validation_data = (
            [x_ab[val_idx_abs], x_ag[val_idx_abs]], y[val_idx_abs],
        )
        print(f"Validation set: {len(val_idx_abs)} pair(s), "
              f"positive rate={float(y[val_idx_abs].mean()):.2f} "
              "(must contain both classes for AUC to mean anything).")
    else:
        validation_data = None

    # Shuffle what is left, so each batch sees both classes rather than a run
    # of 150 consecutive positives followed by batches of pure negatives.
    train_idx = np.random.default_rng(args.seed).permutation(train_idx)

    x_train_ab, x_test_ab = x_ab[train_idx], x_ab[test_idx]
    x_train_ag, x_test_ag = x_ag[train_idx], x_ag[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Class weights, so a ratio above 1:1 does not simply teach the model to
    # answer "no interaction" to everything.
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    class_weight = {0: (pos + neg) / (2 * neg), 1: (pos + neg) / (2 * pos)}
    print(f"Class weights: {{0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}}}")

    input_dimensions = (vector_size, alphabet_size)
    # Names follow the fit() order below ([x_train_ab, x_train_ag]); they used to
    # be the other way round, which read as if the sides were swapped.
    seq_input1 = layers.Input(shape=input_dimensions, name="seq_ab")
    seq_input2 = layers.Input(shape=input_dimensions, name="seq_ag")
    siamese = SiameseInteractionClassifier(
        args.filters, seq_input1, seq_input2,
        embed_dim=args.embed_dim, dropout=args.dropout, learning_rate=args.learning_rate,
    )
    print(f"Model: {siamese.model.count_params():,} parameters, "
          f"{len(x_train_ab):,} training pairs, {len(y_test):,} held-out test pairs.")

    SIAMESE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = siamese_weights_path(vector_size)
    # Monitor val_auc, not val_loss or the old (broken) val_mcc: AUC is
    # threshold-free, so it tracks whether the model is separating the classes
    # at all rather than how confidently it hedges. `restore_best_weights`
    # matters too -- without it the in-memory model used for calibration below
    # is whatever the last epoch happened to produce, not the best one.
    monitor = args.monitor
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(out_path), monitor=monitor, mode="max", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=args.patience, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.5, patience=max(args.patience // 3, 2),
            min_lr=1e-6, verbose=1,
        ),
        EpochStats(args.epochs, monitor=monitor),
    ]

    print(f"Training for up to {args.epochs} epoch(s), batch size {args.batch_size}...")
    history = siamese.model.fit(
        [x_train_ab, x_train_ag], y_train, epochs=args.epochs,
        callbacks=callbacks,
        batch_size=args.batch_size, verbose=1, validation_data=validation_data,
        class_weight=class_weight,
    )

    # return_dict, because Keras 3's `metrics_names` on a compiled model is
    # just ["loss", "compile_metrics"] -- zipping it against the values
    # printed one metric and silently dropped accuracy, AUC, f1 and mcc.
    test_metrics = siamese.model.evaluate(
        [x_test_ab, x_test_ag], y_test, verbose=0, return_dict=True
    )
    print("\nHeld-out test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name:<12} {float(value): .4f}")
    print(f"Best model (by val_auc) saved to {out_path}")

    calibration = write_calibration(
        siamese.model, x_test_ab, x_test_ag, y_test,
        siamese_calibration_path(vector_size), vector_size, args.batch_size,
        validation_data=validation_data,
    )
    d = calibration["decision"]
    print("\nDecision quality on held-out pairs (threshold chosen on validation):")
    for label, row in (("at 0.5", d["at_0.5"]), ("tuned for MCC", d["tuned_for_mcc"]),
                       ("tuned for F1", d["tuned_for_f1"])):
        print(f"  {label:<14} cut {row['threshold']:.3f}  MCC {row['mcc']:+.4f}  "
              f"F1 {row['f1']:.4f}  P {row['precision']:.3f}  R {row['recall']:.3f}")

    print("\nScore calibration (what the dashboard threshold should expect):")
    print(f"  ROC AUC                {calibration['roc_auc']}")
    print(f"  score range            {calibration['score_min']} .. {calibration['score_max']}")
    print(f"  mean score, pos / neg  {calibration['positive_score_mean']} / "
          f"{calibration['negative_score_mean']}")
    print(f"  recommended threshold  {calibration['recommended_threshold']} "
          f"(precision {calibration['recommended_precision']}) -- balanced classification")
    print(f"  selection threshold    {calibration['selection_threshold']} "
          f"(precision {calibration['selection_precision']}) -- for picking candidates")
    print(f"  written to             {siamese_calibration_path(vector_size)}")
    if calibration["score_max"] < 0.8:
        print("  NOTE: this model never scores above 0.80, so leave the dashboard "
              "threshold at or below the recommended value above.")

    plot_history(
        history, PLOTS_DIR / "Siamese_training_curve.png", PLOTS_DIR / "Siamese_training_metrics.png",
    )
    save_history(history, PLOTS_DIR / "Siamese_training_history.json")
    print(f"Saved training curves to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

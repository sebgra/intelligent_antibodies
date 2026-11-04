"""
Siamese antibody/antigen interaction classifier -- the discriminator half of
the pipeline, and the half that decides whether a generated candidate is kept.

Why this was rewritten
----------------------
The previous version trained to a collapsed output: measured on 1500 held-out
equilibrated pairs it scored AUC 0.65, accuracy 0.52 at a 0.5 cut, and -- the
part that broke the dashboard -- every score it produced fell between 0.474
and 0.688 (std 0.036). Generated candidates and *real* antibody chains alike
came back at ~0.55. The dashboard's default 0.80 threshold was therefore
unreachable by construction, so "0 of 20 cleared the threshold" after 800
samples was the only possible outcome; no sampling temperature could fix it.

Three things caused that, all addressed below:

1. The merge was a bare elementwise product of the two towers followed by a
   single sigmoid unit. A product alone cannot express "these two differ",
   and it hands the one output neuron a tiny, poorly-scaled signal -- which
   is what pins the output near its prior. The head now sees the product, the
   absolute difference and the sum, through two hidden layers.
2. No normalization anywhere in a 4-block conv stack feeding a GRU, trained
   at lr 1e-4. Activations drifted and learning was far too slow to escape
   the mean predictor within the early-stopping patience. Every block is now
   BatchNorm'd and the default learning rate is 10x higher.
3. The BiGRU read only the last timestep of a sequence already pooled 16x.
   It now returns the full sequence and is pooled by average *and* max, so
   a local binding motif anywhere along the chain survives to the head.

The architecture stays a true Siamese one: `siamese_propagation` is built
once and applied to both inputs, so both chains are embedded by the *same*
weights.
"""
from keras import layers, Model
from keras.optimizers import Adam
import keras
import keras.ops as ops
from keras.backend import epsilon

# The loss and the accuracy metric that used to live here were hand-rolled
# duplicates of `keras.losses.binary_crossentropy` / `keras.metrics.BinaryAccuracy`,
# so they're gone -- `compile()` below uses the built-ins, which get the shapes
# right on their own. `f1`/`mcc` are kept because Keras 3 has no MCC metric.


@keras.saving.register_keras_serializable(package="intelligent_antibodies")
class ConfusionMetric(keras.metrics.Metric):
        """
        Base class for metrics that must be computed over a WHOLE epoch.

        This exists because passing a plain function as a Keras metric is
        wrong for anything shaped like a ratio of sums. Keras calls such a
        function once per batch and averages the results, and the mean of
        per-batch ratios is not the ratio over the dataset. Measured on 3,321
        held-out pairs with the same predictions:

            global MCC (correct)      +0.2895     F1 0.4855
            batch-averaged, bs=64     +0.0028     F1 0.1804

        So the model looked like it had learned nothing (`val_mcc +0.000`
        epoch after epoch) while it was in fact separating the classes
        perfectly respectably. Accumulating the four confusion counts as
        metric state and dividing once, at `result()`, reports the real
        number -- and early stopping can then safely watch it.
        """

        def __init__(self, threshold=0.5, name="confusion", **kwargs):
                super().__init__(name=name, **kwargs)
                self.threshold = threshold
                self.tp = self.add_variable(shape=(), initializer="zeros", name="tp")
                self.tn = self.add_variable(shape=(), initializer="zeros", name="tn")
                self.fp = self.add_variable(shape=(), initializer="zeros", name="fp")
                self.fn = self.add_variable(shape=(), initializer="zeros", name="fn")

        def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = ops.reshape(ops.cast(y_true, "float32"), (-1,))
                y_pred = ops.reshape(ops.cast(y_pred, "float32"), (-1,))
                pos_pred = ops.cast(y_pred >= self.threshold, "float32")
                pos_true = ops.cast(y_true >= 0.5, "float32")
                neg_pred, neg_true = 1.0 - pos_pred, 1.0 - pos_true
                if sample_weight is not None:
                        w = ops.reshape(ops.cast(sample_weight, "float32"), (-1,))
                else:
                        w = ops.ones_like(y_true)
                self.tp.assign(self.tp + ops.sum(pos_true * pos_pred * w))
                self.tn.assign(self.tn + ops.sum(neg_true * neg_pred * w))
                self.fp.assign(self.fp + ops.sum(neg_true * pos_pred * w))
                self.fn.assign(self.fn + ops.sum(pos_true * neg_pred * w))

        def reset_state(self):
                for v in (self.tp, self.tn, self.fp, self.fn):
                        v.assign(ops.zeros(()))

        def get_config(self):
                return {**super().get_config(), "threshold": self.threshold}


@keras.saving.register_keras_serializable(package="intelligent_antibodies")
class MCCMetric(ConfusionMetric):
        """Matthews correlation coefficient over the whole epoch."""

        def __init__(self, threshold=0.5, name="mcc", **kwargs):
                super().__init__(threshold=threshold, name=name, **kwargs)

        def result(self):
                num = self.tp * self.tn - self.fp * self.fn
                den = ops.sqrt((self.tp + self.fp) * (self.tp + self.fn)
                               * (self.tn + self.fp) * (self.tn + self.fn))
                return num / (den + epsilon())


@keras.saving.register_keras_serializable(package="intelligent_antibodies")
class F1Metric(ConfusionMetric):
        """F1 over the whole epoch."""

        def __init__(self, threshold=0.5, name="f1", **kwargs):
                super().__init__(threshold=threshold, name=name, **kwargs)

        def result(self):
                precision = self.tp / (self.tp + self.fp + epsilon())
                recall = self.tp / (self.tp + self.fn + epsilon())
                return 2 * precision * recall / (precision + recall + epsilon())


def _flatten(y_true, y_pred):
        """
        Collapse labels and predictions to matching 1-D tensors.

        Keras hands these over as `(batch,)` labels against a `(batch, 1)`
        sigmoid output. Multiplying them elementwise therefore broadcasts into
        a `(batch, batch)` OUTER PRODUCT, and every count below turns into a
        sum over batch^2 cross terms instead of per-sample pairs. That silently
        pins `accuracy` to exactly `q*p + (1-q)*(1-p)` -- i.e. 0.50 on a
        balanced set, whatever the model predicts -- forces `mcc` to an
        identical 0, and lets `f1` climb past 1. Reshape first, always.
        """
        return ops.reshape(y_true, (-1,)), ops.reshape(y_pred, (-1,))


def f1(y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        tp = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
        possible_positives = ops.sum(ops.round(ops.clip(y_true, 0, 1)))
        pred_pos = ops.sum(ops.round(ops.clip(y_pred, 0, 1)))
        precision = tp / (pred_pos + epsilon())
        recall = tp / (possible_positives + epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+epsilon())
        return f1_val


def mcc(y_true, y_pred):
        y_true, y_pred = _flatten(y_true, y_pred)
        y_pred_pos = ops.round(ops.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = ops.round(ops.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tp = ops.sum(y_pos * y_pred_pos)
        tn = ops.sum(y_neg * y_pred_neg)
        fp = ops.sum(y_neg * y_pred_pos)
        fn = ops.sum(y_pos * y_pred_neg)
        numerator = (tp * tn - fp * fn)
        denominator = ops.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / (denominator + epsilon())



class SiameseInteractionClassifier:

        def __init__(
                self,
                filters,
                seq_input1,
                seq_input2,
                embed_dim: int = 32,
                dropout: float = 0.3,
                learning_rate: float = 1e-3,
                label_smoothing: float = 0.02,
        ):
                # A learned projection of the 22-letter one-hot input. One-hot
                # residues are mutually orthogonal, so the first conv layer has to
                # spend capacity discovering that (say) I and V behave alike; a
                # cheap linear embedding lets it start from a similarity space
                # instead. Input/output interface is unchanged -- callers still
                # pass one-hot (vector_size, alphabet_size) tensors.
                self.embed = layers.Dense(embed_dim, use_bias=False, name="residue_embedding")

                # Pool by 2, not 3. Four stride-3 pools divide the sequence by 81,
                # which left the BiGRU below reading 2 timesteps out of a 200-residue
                # input -- there was no sequence left for it to model. /2 each keeps
                # 1/16th of the length (400 -> 25), which the GRU can actually use.
                #
                # Each block is Conv -> BatchNorm -> ReLU -> MaxPool. The missing
                # normalization was a direct cause of the collapsed output: with
                # four unnormalized conv layers feeding a GRU at lr 1e-4, training
                # early-stopped long before the head learned anything but the mean.
                self.conv01 = layers.Conv1D(filters, 11, padding='same')
                self.bn01 = layers.BatchNormalization()
                self.mp1 = layers.MaxPooling1D(2)
                self.conv02 = layers.Conv1D(filters*2, 7, padding='same')
                self.bn02 = layers.BatchNormalization()
                self.mp2 = layers.MaxPooling1D(2)
                self.conv03 = layers.Conv1D(filters*4, 3, padding='same')
                self.bn03 = layers.BatchNormalization()
                self.mp3 = layers.MaxPooling1D(2)
                self.conv04 = layers.Conv1D(filters*2, 3, padding='same')
                self.bn04 = layers.BatchNormalization()
                self.mp4 = layers.MaxPooling1D(2)
                self.act = layers.Activation("relu")

                # return_sequences=True + dual pooling. Reading only the GRU's final
                # state discards where along the chain a binding motif sat; average
                # and max pooling over all timesteps keep both the overall profile
                # and the single strongest local response.
                self.gru = layers.Bidirectional(
                        layers.GRU(filters, return_sequences=True), name="bigru"
                )
                self.gap = layers.GlobalAveragePooling1D()
                self.gmp = layers.GlobalMaxPooling1D()

                # Interaction head. `multiply` alone was the old merge; it cannot
                # represent dissimilarity, so |left - right| and (left + right) go
                # in alongside it, and two hidden layers sit between the merge and
                # the sigmoid instead of a single unit.
                self.head_norm = layers.BatchNormalization(name="merge_norm")
                self.head_dense1 = layers.Dense(256, activation="relu", name="head_1")
                self.head_drop1 = layers.Dropout(dropout)
                self.head_dense2 = layers.Dense(64, activation="relu", name="head_2")
                self.head_drop2 = layers.Dropout(dropout)
                self.head_out = layers.Dense(1, activation="sigmoid", name="interaction")

                self.model = Model(inputs=[seq_input1, seq_input2],
                outputs=[self.forward(seq_input1, seq_input2)])
                # amsgrad kept; clipnorm added because the BiGRU occasionally spikes
                # early on, and a touch of label smoothing stops the head from
                # saturating on the easy negatives of a balanced set.
                adam = Adam(learning_rate=learning_rate, amsgrad=True, epsilon=1e-6, clipnorm=1.0)
                self.model.compile(
                        optimizer=adam,
                        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                        # AUC is threshold-free, so a model that has collapsed to a
                        # constant output shows up immediately as ~0.5 instead of
                        # hiding behind an accuracy that looks like chance anyway.
                        # MCC and F1 are the stateful classes above, NOT the bare
                        # functions: passed as functions Keras averages them per
                        # batch, which reported +0.000 for a model whose real MCC
                        # was +0.29.
                        metrics=[
                                "accuracy",
                                keras.metrics.AUC(name="auc"),
                                keras.metrics.Precision(name="precision"),
                                keras.metrics.Recall(name="recall"),
                                F1Metric(name="f1"),
                                MCCMetric(name="mcc"),
                        ],
                )

        def siamese_propagation(self, x):
                x = self.embed(x)

                x = self.act(self.bn01(self.conv01(x)))
                x = self.mp1(x)

                x = self.act(self.bn02(self.conv02(x)))
                x = self.mp2(x)

                x = self.act(self.bn03(self.conv03(x)))
                x = self.mp3(x)

                x = self.act(self.bn04(self.conv04(x)))
                x = self.mp4(x)

                x_gru = self.gru(x)
                return layers.concatenate([self.gap(x_gru), self.gmp(x_gru)])

        def forward(self, left, right):
                left = self.siamese_propagation(left)
                right = self.siamese_propagation(right)

                merge = layers.concatenate([
                        layers.multiply([left, right]),
                        layers.subtract([left, right]),
                        layers.add([left, right]),
                ])
                merge = self.head_norm(merge)
                merge = self.head_drop1(self.head_dense1(merge))
                merge = self.head_drop2(self.head_dense2(merge))
                return self.head_out(merge)


if __name__ == "__main__":
        import numpy as np

        # Self-check: metrics must survive Keras' real shapes -- `(batch,)` labels
        # against a `(batch, 1)` sigmoid output. Passing these unflattened
        # broadcasts to `(batch, batch)`, which drives f1 above 1 and pins mcc to
        # 0 on a balanced set regardless of how good the prediction is.
        y_true = np.array([1.0, 1.0, 0.0, 0.0])              # (4,)
        perfect = np.array([[0.99], [0.98], [0.02], [0.01]])  # (4, 1)
        inverted = 1.0 - perfect

        assert abs(float(f1(y_true, perfect)) - 1.0) < 1e-3, float(f1(y_true, perfect))
        assert abs(float(mcc(y_true, perfect)) - 1.0) < 1e-3, float(mcc(y_true, perfect))
        assert abs(float(f1(y_true, inverted))) < 1e-3, float(f1(y_true, inverted))
        assert abs(float(mcc(y_true, inverted)) + 1.0) < 1e-3, float(mcc(y_true, inverted))

        # Self-check: pooling must leave the BiGRU an actual sequence to read.
        # Four stride-3 pools reduced a 400-residue input to 4 timesteps.
        vector_size, alphabet_size = 400, 22
        a = layers.Input(shape=(vector_size, alphabet_size))
        b = layers.Input(shape=(vector_size, alphabet_size))
        model = SiameseInteractionClassifier(8, a, b).model
        gru_steps = next(
            l.input.shape[1] for l in model.layers if isinstance(l, layers.Bidirectional)
        )
        assert gru_steps >= 16, f"BiGRU sees only {gru_steps} timestep(s)"

        # Self-check: the head must be able to tell a pair from its own mirror.
        # With the old bare-product merge the model was symmetric by
        # construction -- f(a, b) == f(b, a) -- so no amount of training could
        # separate an (antibody, antigen) pair from the reversed pairing.
        x1 = np.random.default_rng(0).random((4, vector_size, alphabet_size)).astype("float32")
        x2 = np.random.default_rng(1).random((4, vector_size, alphabet_size)).astype("float32")
        forward_scores = model.predict([x1, x2], verbose=0)
        mirror_scores = model.predict([x2, x1], verbose=0)
        assert not np.allclose(forward_scores, mirror_scores), (
            "merge is still symmetric: f(a, b) == f(b, a)"
        )

        print("SiameseInteractionClassifier self-check OK")


def _selfcheck_stateful_metrics():
        """Stateful metrics must match sklearn over the whole set, batched or not."""
        import numpy as np
        from sklearn.metrics import f1_score, matthews_corrcoef

        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.27).astype("float32")
        p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.22, 2000), 0, 1).astype("float32")

        for metric, reference in ((MCCMetric(), matthews_corrcoef(y, p >= 0.5)),
                                  (F1Metric(), f1_score(y, p >= 0.5))):
                metric.reset_state()
                for i in range(0, len(y), 64):           # fed batch by batch, as Keras does
                        metric.update_state(y[i:i + 64], p[i:i + 64].reshape(-1, 1))
                got = float(metric.result())
                assert abs(got - reference) < 1e-4, f"{metric.name}: {got} vs {reference}"
        print("stateful metric self-check OK (batched == global)")


if __name__ == "__main__":
        _selfcheck_stateful_metrics()

from keras import layers, Model
from keras.optimizers import Adam
import keras
import keras.ops as ops
from keras.backend import epsilon

# The loss and the accuracy metric that used to live here were hand-rolled
# duplicates of `keras.losses.binary_crossentropy` / `keras.metrics.BinaryAccuracy`,
# so they're gone -- `compile()` below uses the built-ins, which get the shapes
# right on their own. `f1`/`mcc` are kept because Keras 3 has no MCC metric.


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

        def __init__(self, filters, seq_input1, seq_input2):

                # Pool by 2, not 3. Four stride-3 pools divide the sequence by 81,
                # which left the BiGRU below reading 2 timesteps out of a 200-residue
                # input -- there was no sequence left for it to model. /2 each keeps
                # 1/16th of the length (400 -> 25), which the GRU can actually use.
                self.conv01 = layers.Conv1D(filters, 11, padding='same', activation="relu")
                self.mp1 = layers.MaxPooling1D(2)
                self.conv02 = layers.Conv1D(filters*2, 7, padding='same', activation="relu")
                self.mp2 = layers.MaxPooling1D(2)
                self.conv03 = layers.Conv1D(filters*4, 3, padding='same', activation="relu")
                self.mp3 = layers.MaxPooling1D(2)
                self.conv04 = layers.Conv1D(filters*2, 3, padding='same', activation="relu")
                self.mp4 = layers.MaxPooling1D(2)

                self.gru = layers.Bidirectional(layers.GRU(filters, return_sequences=False))

                self.model = Model(inputs=[seq_input1, seq_input2],
                outputs=[self.forward(seq_input1, seq_input2)])
                adam = Adam(learning_rate=1e-4, amsgrad=True, epsilon=1e-6)
                self.model.compile(optimizer=adam, loss="binary_crossentropy",
                                   metrics=["accuracy", f1, mcc])

        def siamese_propagation(self, x):
                x = self.conv01(x)
                x = self.mp1(x)

                x = self.conv02(x)
                x = self.mp2(x)

                x = self.conv03(x)
                x = self.mp3(x)

                x = self.conv04(x)
                x = self.mp4(x)

                x_gru = self.gru(x)
                return x_gru

        def forward(self, left, right):
                left = self.siamese_propagation(left)
                right = self.siamese_propagation(right)

                merge = layers.multiply([left, right])
                merge = layers.Dropout(0.2)(merge)
                return layers.Dense(1, activation='sigmoid')(merge)


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

        print("SiameseInteractionClassifier self-check OK")


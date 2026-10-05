from keras import layers, Model
from keras.optimizers import Adam
import keras
import keras.ops as ops
from keras.backend import epsilon

# NOTE: these custom metrics/loss were originally written against the Keras 2
# `keras.backend` API (`K.clip`, `K.round`, `K.sum`, `K.sqrt`, `K.mean`, `K.log`),
# which no longer exists on Keras 3 (only `epsilon()` survived on `backend`).
# They're rewritten here against `keras.ops`, Keras 3's backend-agnostic tensor
# API, which is the direct replacement -- the math is unchanged.


def accuracy(y_true, y_pred):
        y_pred_pos = ops.round(ops.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = ops.round(ops.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tp = ops.sum(y_pos * y_pred_pos)
        tn = ops.sum(y_neg * y_pred_neg)
        fp = ops.sum(y_neg * y_pred_pos)
        fn = ops.sum(y_pos * y_pred_neg)
        return (tp + tn) / (tp + tn + fp + fn)

def binary_crossentropy(y_true, y_pred):
        y_pred = ops.clip(y_pred, epsilon(), 1 - epsilon())
        loss = - ops.mean(y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
        return loss


def f1(y_true, y_pred):
        tp = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
        possible_positives = ops.sum(ops.round(ops.clip(y_true, 0, 1)))
        pred_pos = ops.sum(ops.round(ops.clip(y_pred, 0, 1)))
        precision = tp / (pred_pos + epsilon())
        recall = tp / (possible_positives + epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+epsilon())
        return f1_val


def mcc(y_true, y_pred):
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

                self.conv01 = layers.Conv1D(filters, 11, padding='same', activation="relu")
                self.mp1 = layers.MaxPooling1D(3)
                self.conv02 = layers.Conv1D(filters*2, 7, padding='same', activation="relu")
                self.mp2 = layers.MaxPooling1D(3)
                self.conv03 = layers.Conv1D(filters*4, 3, padding='same', activation="relu")
                self.mp3 = layers.MaxPooling1D(3)
                self.conv04 = layers.Conv1D(filters*2, 3, padding='same', activation="relu")
                self.mp4 = layers.MaxPooling1D(3)

                self.gru = layers.Bidirectional(layers.GRU(filters, return_sequences=False))
                
                self.model = Model(inputs=[seq_input1, seq_input2],
                outputs=[self.forward(seq_input1, seq_input2)])
                adam = Adam(learning_rate=1e-4, amsgrad=True, epsilon=1e-6)
                self.model.compile(optimizer=adam, loss=binary_crossentropy, metrics=[accuracy, f1, mcc])

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

